import os
import sys
import copy
from packaging import version
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class VAE_encode(nn.Module):
    def __init__(self, vae):
        super(VAE_encode, self).__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae):
        super(VAE_decode, self).__init__()
        self.vae = vae

    def forward(self, x):
        assert self.vae.encoder.current_down_blocks is not None
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        return self.vae.decoder(x / self.vae.config.scaling_factor).clamp(-1, 1)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_weights(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        # feats is a list of features form different layers
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            b, h, w = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                mlp.to(feat.device)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([b, x_sample.shape[-1], h, w])
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    def __init__(self, batch_size, num_patches, nce_T=0.07, nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.nce_T = nce_T
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches_per_batch = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches_per_batch, 1, -1), feat_k.view(num_patches_per_batch, -1, 1))
        l_pos = l_pos.view(num_patches_per_batch, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


class CUT_turbo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)
        self.sched = make_1step_sched()
        if args.pretrained_path == None:
            self.vae, self.vae_lora_target_modules \
                = self.initialize_vae(args.lora_rank_vae, return_lora_module_names=True)
            self.unet, self.l_modules_unet_encoder, self.l_modules_unet_decoder, self.l_modules_unet_others \
                = self.initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
            self.vae_enc = VAE_encode(self.vae)
            self.vae_dec = VAE_decode(self.vae)
        else:
            sd = torch.load(args.pretrained_path)
            self.vae, self.vae_enc, self.vae_dec, self.vae_lora_target_modules \
                = self.load_vae_from_state_dict(sd, return_lora_module_names=True)
            self.unet, self.l_modules_unet_encoder, self.l_modules_unet_decoder, self.l_modules_unet_others \
                = self.load_unet_from_state_dict(sd, return_lora_module_names=True)

    @staticmethod
    def initialize_vae(rank=4, return_lora_module_names=False):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.requires_grad_(False)
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.requires_grad_(True)
        vae.train()
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1
        # the modules of vae add lora adapters
        l_vae_target_modules = ["conv1", "conv2", "conv_in", "conv_shortcut",
                                "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3",
                                "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
                                ]
        vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        if return_lora_module_names:
            return vae, l_vae_target_modules
        else:
            return vae

    @staticmethod
    def initialize_unet(rank = 8, return_lora_module_names=False):
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        unet.requires_grad_(False)
        unet.train()
        l_target_modules_encoder, l_target_modules_decoder, l_target_modules_others = [], [], []
        # the modules of unet add lora adapters
        l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out",
                  "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
        for n, p in unet.named_parameters():
            if "bias" in n or "norm" in n: continue
            for pattern in l_grep:
                if pattern in n and ("down_blocks" in n or "conv_in" in n):
                    l_target_modules_encoder.append(n.replace(".weight",""))
                    break
                elif pattern in n and "up_blocks" in n:
                    l_target_modules_decoder.append(n.replace(".weight",""))
                    break
                elif pattern in n:
                    l_target_modules_others.append(n.replace(".weight",""))
                    break
        lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder, lora_alpha=rank)
        lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder, lora_alpha=rank)
        lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_others, lora_alpha=rank)
        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        if return_lora_module_names:
            return unet, l_target_modules_encoder, l_target_modules_decoder, l_target_modules_others
        else:
            return unet

    @staticmethod
    def load_vae_from_state_dict(sd, return_lora_module_names=False):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.requires_grad_(False)
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.requires_grad_(True)
        vae.train()
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1),
                                                  bias=False).requires_grad_(True)
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1
        l_vae_target_modules = sd["vae_lora_target_modules"]
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        vae_enc = VAE_encode(vae)
        vae_enc.load_state_dict(sd["sd_vae_enc"])
        vae_dec = VAE_decode(vae)
        vae_dec.load_state_dict(sd["sd_vae_dec"])

        if return_lora_module_names:
            return vae, vae_enc, vae_dec, l_vae_target_modules
        else:
            return vae, vae_enc, vae_dec

    @staticmethod
    def load_unet_from_state_dict(sd, return_lora_module_names=False):
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        unet.requires_grad_(False)
        unet.train()
        l_target_modules_encoder = sd["l_target_modules_encoder"]
        l_target_modules_decoder = sd["l_target_modules_decoder"]
        l_target_modules_others = sd["l_target_modules_others"]
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=sd["l_target_modules_others"], lora_alpha=sd["rank_unet"])
        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_others"][name_sd])
        unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        if return_lora_module_names:
            return unet, l_target_modules_encoder, l_target_modules_decoder, l_target_modules_others
        else:
            return unet

    def get_trainable_params(self):
        # add unet parameters
        params_gen = list(self.unet.conv_in.parameters())
        self.unet.conv_in.requires_grad_(True)
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        for n,p in self.unet.named_parameters():
            if "lora" in n and "default" in n:
                assert p.requires_grad
                params_gen.append(p)

        # add vae parameters
        for n, p in self.vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(self.vae.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(self.vae.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(self.vae.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(self.vae.decoder.skip_conv_4.parameters())
        return params_gen

    def forward(self, x, caption=None, caption_emb=None):
        assert (caption is not None or caption_emb is not None)
        B = x.shape[0]
        timesteps = torch.tensor([self.sched.config.num_train_timesteps - 1] * B,
                                 device=x.device).long()
        x_enc = self.vae_enc(x).to(x.dtype)
        if caption_emb is None:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x.device)
            caption_emb = self.text_encoder(caption_tokens)[0].detach().clone()
        model_pred = self.unet(x_enc, timesteps, encoder_hidden_states=caption_emb).sample
        x_out = torch.stack([self.sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = self.vae_dec(x_out)
        return x_out_decoded


if __name__ == '__main__':
    feats = []
    featsq = []
    total_nce_loss = 0.0
    crit_contrastive = PatchNCELoss(4, 8)
    feat1 = torch.rand([4, 3, 16, 16])
    feat2 = torch.rand([4, 6, 8, 8])
    feats.append(feat1)
    feats.append(feat2)
    feat1q = torch.rand([4, 3, 16, 16])
    feat2q = torch.rand([4, 6, 8, 8])
    featsq.append(feat1q)
    featsq.append(feat2q)
    moduleF = PatchSampleF(use_mlp=True)
    feat_pool, sample_ids = moduleF(feats, num_patches=8)
    featq_pool, _ = moduleF(featsq, 8, sample_ids)
    for feat in feat_pool:
        print(feat.shape)
    for feat, featq in zip(feat_pool, featq_pool):
        total_nce_loss += crit_contrastive(feat, featq)
    total_nce_loss /= 2
    print(total_nce_loss)