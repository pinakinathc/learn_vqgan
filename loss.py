"""
Self-contained Minimal implementation of VQ-GAN model.
Paper: https://arxiv.org/abs/2012.09841
Reference: https://compvis.github.io/taming-transformers
Copyright: Do whatever you want. Don't ask me.

"""

import requests
import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0)


def hinge_disc_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    disc_loss = 0.5 * (loss_real + loss_fake)
    return disc_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self, disc_start=10000, codebook_weight=1.0, pixelloss_weight=1.0,
        disc_num_layers=3, disc_in_channel=3, disc_factor=1.0, disc_weight=0.8,
        perceptual_weight=1.0, conv_filters=64, device=torch.device("cuda:0")
    ):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_loss = self.perceptual_loss.to(device)
        self.perceptual_weight = perceptual_weight
        self.discriminator = DiscriminatorPatchGAN(
            input_channel=disc_in_channel, num_layers=disc_num_layers,
            conv_filters=conv_filters).apply(weights_init).to(device)
        self.disc_iter_start = disc_start
        self.disc_loss = hinge_disc_loss
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimiser_idx, global_step, last_layer, split="train"):
        rec_loss = torch.abs(inputs - reconstructions)
        percep_loss = self.perceptual_loss(inputs, reconstructions)
        rec_loss = rec_loss + self.perceptual_weight * percep_loss
        nll_loss = torch.mean(rec_loss)

        # Generator Update in GAN
        if optimiser_idx == "generator":
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -1 * torch.mean(logits_fake)
            adaptive_weight = self.calculate_adaptive_weight(
                nll_loss, generator_loss, last_layer=last_layer
            )
            disc_factor = 0.0 if global_step < self.disc_iter_start else self.disc_factor
            loss = nll_loss + adaptive_weight * disc_factor * generator_loss + self.codebook_weight * codebook_loss.mean()

            loss_info = {
                f"{split}/total_loss_step": loss.clone().detach().mean(),
                f"{split}/quant_loss_step": codebook_loss.detach().mean(),
                f"{split}/nll_loss_step": nll_loss.detach().mean(),
                f"{split}/rec_loss_step": rec_loss.detach().mean(),
                f"{split}/p_loss_step": percep_loss.detach().mean(),
                f"{split}/d_weight_step": adaptive_weight.detach(),
                f"{split}/disc_factor_step": torch.tensor(disc_factor),
                f"{split}/g_loss_step": generator_loss.detach().mean()
            }
            return loss, loss_info

        # Discriminator Update in GAN
        if optimiser_idx == "discriminator":
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())
            disc_factor = 0.0 if global_step < self.disc_iter_start else self.disc_factor
            loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            loss_info = {
                f"{split}/disc_loss_step": loss.clone().detach().mean(),
                f"{split}/logits_real_step": logits_real.detach().mean(),
                f"{split}/logits_fake_step": logits_fake.detach().mean()
            }
            return loss, loss_info

    def calculate_adaptive_weight(self, nll_loss, generator_loss, last_layer):
        nll_grad = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        generator_grad = torch.autograd.grad(generator_loss, last_layer, retain_graph=True)[0]
        adaptive_weight = torch.norm(nll_grad) / (torch.norm(generator_grad) + 1e-4)
        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
        adaptive_weight = adaptive_weight * self.disc_weight
        return adaptive_weight


class DiscriminatorPatchGAN(nn.Module):
    """
    Defines a PatchGAN discriminator as in Pix2Pix.
    Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    """
    def __init__(self, input_channel=3, conv_filters=64, num_layers=3):
        super().__init__()
        sequence = [
            nn.Conv2d(input_channel, conv_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True) # in-place operation. Saves memory/bandwidth
        ]
        filter_mult = 1
        filter_mult_prev = 1
        for n in range(1, num_layers): # gradually increase the number of filters
            filter_mult_prev = filter_mult
            filter_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(conv_filters * filter_mult_prev, conv_filters * filter_mult, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(conv_filters * filter_mult),
                nn.LeakyReLU(0.2, True) # in-place operation
            ]

        filter_mult_prev = filter_mult
        filter_mult = min(2**num_layers, 8)
        sequence += [
            nn.Conv2d(conv_filters * filter_mult_prev, conv_filters * filter_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_filters * filter_mult),
            nn.LeakyReLU(0.2, True) # in-place
        ]

        sequence += [
            nn.Conv2d(conv_filters * filter_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs):
        return self.main(inputs)


class LPIPS(nn.Module):
    """
    Learned perceptual metric.

    """
    def __init__(self):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512] # VGG16 Features
        self.net = VGG16()
        self.lin0 = NetLinLayer(self.channels[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.channels[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.channels[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.channels[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.channels[4], use_dropout=True)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        # Download ckpt
        URL = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
        chunk_size = 1024
        with requests.get(URL, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                with open("./vgg.pth", "wb") as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)
        
        self.load_state_dict(torch.load("./vgg.pth", map_location=torch.device("cpu")), strict=False)
        print("Loaded pretrained LPIPS loss from vgg.pth")

    def forward(self, inputs, target):
        in0_input, in1_input = self.scaling_layer(inputs), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for idx in range(len(self.channels)):
            # Normalise tensor
            feats0[idx] = outs0[idx] / (torch.sqrt(torch.sum(outs0[idx] ** 2, dim=1, keepdim=True)) + 1e-10)
            feats1[idx] = outs1[idx] / (torch.sqrt(torch.sum(outs1[idx] ** 2, dim=1, keepdim=True)) + 1e-10)
            diffs[idx]  = (feats0[idx] - feats1[idx]) ** 2

        res = []
        res.append(
            (self.lin0.model(diffs[0])).mean([2, 3], keepdim=True) # Spatial Average
        )
        res.append((self.lin1.model(diffs[1])).mean([2, 3], keepdim=True))
        res.append((self.lin2.model(diffs[2])).mean([2, 3], keepdim=True))
        res.append((self.lin3.model(diffs[3])).mean([2, 3], keepdim=True))
        res.append((self.lin4.model(diffs[4])).mean([2, 3], keepdim=True))

        val = res[0]
        for idx in range(1, len(self.channels)):
            val += res[idx]
        return val


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.30, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


if __name__ == "__main__":
    torch.manual_seed(1234)
    lpips = LPIPS().eval()

    for i in range(10):
        inp   = torch.randn(4, 3, 256, 256)
        recon = torch.randn(4, 3, 256, 256)
        loss = lpips(inp, recon)
        print (i, loss)
