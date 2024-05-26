import os
import gc
import lpips
import torch
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import transformers

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import vision_aided_loss

from CUT_turbo import CUT_turbo, PatchSampleF, PatchNCELoss
from utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_contrastive_training
from utils.dino_struct import DinoStructureLoss


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_gen = CUT_turbo(args)

    if args.gan_disc_type == "vagan_clip":
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc.train()
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    module_f = PatchSampleF(args.use_mlp, args.init_type, nc=args.vector_nc)

    crit_idt = torch.nn.L1Loss()
    crit_contrastive = PatchNCELoss(args.train_batch_size, args.num_patches, args.nce_T, args.nce_includes_all_negatives_from_minibatch)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_gen.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_gen.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_gen.unet.conv_in.requires_grad_(True)
    params_gen = net_gen.get_trainable_params()
    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                      weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    params_disc = list(net_disc.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    # input-dependent initialization of mlp in module_f
    nce_layers_list = [int(i) for i in args.nce_layers.split(',')]
    n_layers = len(nce_layers_list)
    if args.use_mlp:
        _, feat_list_entire = net_gen.vae.encoder(torch.rand([1, 3, 512, 512]), inter_feat=True)
        feat_list = []
        for i in nce_layers_list:
            feat_list.append(feat_list_entire[i])
        _, _ = module_f(feat_list, args.num_patches, None)
        params_contrastive = list(module_f.parameters())
        optimizer_contrastive = torch.optim.AdamW(params_contrastive, lr=args.learning_rate,
                                                betas=(args.adam_beta1, args.adam_beta2),
                                                weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train",
                                    tokenizer=net_gen.tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                   num_workers=args.dataloader_num_workers)

    # images for val
    T_val = build_transform(args.val_img_prep)
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    # make the reference FID statistics
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        output_ref = os.path.join(args.output_dir, "fid_reference")
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.join(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_ref, model=feat_model, num_workers=0, num=None,
                                           shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                           mode="clean", custom_fn_resize=None, description="", verbose=True,
                                           custom_image_tranform=None)
    # TODO: save scheduler to checkpoint
    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps * accelerator.num_processes,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)
    if args.use_mlp:
        lr_scheduler_contrastive = get_scheduler(args.lr_scheduler, optimizer=optimizer_contrastive,
                                                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                                num_training_steps=args.max_train_steps * accelerator.num_processes,
                                                num_cycles=args.lr_num_cycles, power=args.lr_power)
    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.requires_grad_(False)

    # Prepare everything with our 'accelerator'.
    # net_gen, net_disc, module_f = accelerator.prepare(net_gen, net_disc, module_f)
    net_gen.unet, net_gen.vae_enc, net_gen.vae_dec = accelerator.prepare(net_gen.unet, net_gen.vae_enc, net_gen.vae_dec)
    net_disc = accelerator.prepare(net_disc)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen,
        lr_scheduler_disc
    )
    if (args.use_mlp):
        module_f = accelerator.prepare(module_f)
        optimizer_contrastive = accelerator.prepare(optimizer_contrastive)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move all networks to device and cast to weight_dtype
    net_gen.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)),
                                  init_kwargs={"wandb": {"dir": args.logging_dir}})

    fixed_tokens = dataset_train.input_ids_tgt[0]
    fixed_emb_base = net_gen.text_encoder(fixed_tokens.cuda().unsqueeze(0))[0].detach()

    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    # turn off eff. attn for the disc
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # start the training loop
    for epoch in range(args.first_train_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [net_gen, net_disc, module_f]
            with accelerator.accumulate(*l_acc):
                real_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                real_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                bsz = real_a.shape[0]
                fixed_emb = fixed_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                """
                GAN Objective
                """
                fake_b = net_gen(real_a, caption_emb=fixed_emb)
                loss_G = net_disc(fake_b, for_G=True).mean() * args.lambda_gan
                """
                Contrastive Objective
                """
                fake_b = net_gen(real_a, caption_emb=fixed_emb)
                _, feat_real_a_list_entire = net_gen.vae.encoder(real_a, inter_feat=True)
                _, feat_fake_b_list_entire = net_gen.vae.encoder(fake_b, inter_feat=True)
                feat_real_a_list = []
                feat_fake_b_list = []
                for i in nce_layers_list:
                    feat_real_a_list.append(feat_real_a_list_entire[i])
                    feat_fake_b_list.append(feat_fake_b_list_entire[i])
                feat_k_pool, sample_ids = module_f(feat_real_a_list, args.num_patches, None)
                feat_q_pool, _ = module_f(feat_fake_b_list, args.num_patches)
                total_nce_loss = 0
                for feat_q, feat_k in zip(feat_q_pool, feat_k_pool):
                    nce_loss = crit_contrastive(feat_q, feat_k).mean()
                    total_nce_loss = total_nce_loss + nce_loss
                total_nce_loss = total_nce_loss / n_layers * args.lambda_contrastive
                loss_G_total = loss_G + total_nce_loss
                accelerator.backward(loss_G_total, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                if args.use_mlp:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_contrastive, args.max_grad_norm)
                    optimizer_contrastive.step()
                    lr_scheduler_contrastive.step()
                    optimizer_contrastive.zero_grad()
                """
                Identity Objective
                """
                idt_b = net_gen(real_b, caption_emb=fixed_emb)
                loss_idt = crit_idt(idt_b, real_b) * args.lambda_idt + net_lpips(idt_b, real_b).mean() * args.lambda_idt_lpips
                accelerator.backward(loss_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                """
                Discriminator Objective
                """
                loss_D_fake = net_disc(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(net_disc.parameters()), args.max_grad_norm)
                optimizer_disc.step()
                # lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                loss_D_real = net_disc(real_b, for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(net_disc.parameters()), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            logs["gan"] = loss_G.detach().item()
            logs["disc"] = loss_D_fake.detach().item() + loss_D_real.detach().item()
            logs["idt"] = loss_idt.detach().item()
            logs["contrastive"] = total_nce_loss.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(net_gen.unet)
                    eval_vae_enc = accelerator.unwrap_model(net_gen.vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(net_gen.vae_dec)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            # Now only support wandb
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"]
                                viz_img_b = batch["pixel_values_tgt"]
                                log_dict = {
                                    "train/real_a": [
                                        wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}")
                                        for idx in range(bsz)],
                                    "train/real_b": [
                                        wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}")
                                        for idx in range(bsz)],
                                    "train/fake_b": [
                                        wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}")
                                        for idx in range(bsz)],
                                    "train/idt_b": [
                                        wandb.Image(idt_b[idx].float().detach().cpu(), caption=f"idx={idx}")
                                        for idx in range(bsz)],
                                }
                                log_dict.update(logs)
                                tracker.log(log_dict, step=global_step)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = net_gen.vae_lora_target_modules # list
                        sd["sd_vae_enc"] = net_gen.vae_enc.state_dict()
                        sd["sd_vae_dec"] = net_gen.vae_dec.state_dict()
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["l_target_modules_encoder"] = net_gen.l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = net_gen.l_modules_unet_decoder
                        sd["l_target_modules_others"] = net_gen.l_modules_unet_others
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 1 and args.track_val_fid:
                        timesteps = torch.tensor([net_gen.sched.config.num_train_timesteps - 1] * 1,
                                                  device="cuda").long()
                        net_dino = DinoStructureLoss()
                        """
                        Evaluate
                        """
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores = []
                       # cal dino score
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if idx > args.validation_num_images and args.validation_num_images >= 0:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                eval_real_a = transforms.ToTensor()(input_img)
                                eval_real_a = transforms.Normalize([0.5], [0.5])(eval_real_a)
                                eval_enc = eval_vae_enc(eval_real_a).to(eval_real_a.dtype)
                                eval_model_pred = eval_unet(eval_enc, timesteps, encoder_hidden_states=fixed_emb)
                                eval_out = torch.stack([net_gen.sched.step(eval_model_pred[0], timesteps[0], eval_enc[0], return_dict=True).prev_sample])
                                eval_fake_b = eval_vae_dec(eval_out)
                                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)
                                dino_ssim = net_dino.calculate_global_ssim_loss(
                                    net_dino.preprocess(input_img).unsqueeze(0).cuda(),
                                    net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                ).item()
                                l_dino_scores.append(dino_ssim)
                        dino_score = np.mean(l_dino_scores)
                        # cal FID score
                        eval_fake_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                                                           shuffle=False, seed=0, batch_size=8,
                                                           device=torch.device("cuda"),
                                                           mode="clean", custom_fn_resize=None, description="",
                                                           verbose=True,
                                                           custom_image_tranform=None)
                        fid_score = fid_from_feats(ref_features, eval_fake_features)
                        print(f"step={global_step}, fid(a2b)={fid_score:.2f}, dino(a2b)={dino_score:.3f}")

                        logs["val/dino_struct"] = dino_score
                        logs["val/fid"] = fid_score
                        del net_dino

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                accelerator.end_training()
                break
    accelerator.end_training()
if __name__ == '__main__':
    args = parse_args_unpaired_contrastive_training()
    main(args)
