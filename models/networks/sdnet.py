import torch
import torch.nn as nn
from models.layers.blocks import Interpolate, ResConv, conv_block_unet, conv_bn_lrelu, upconv


class AEncoder(nn.Module):
    def __init__(self, in_channel, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.in_channel = in_channel
        self.width = width
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(self.in_channel, self.width, self.height, self.ndf, self.num_output_channels, self.norm, self.upsample)

    def forward(self, x):
        out = self.unet(x)
        out = torch.tanh(out)

        return out


class MEncoder(nn.Module):
    def __init__(self, z_length, in_channel, img_size):
        super(MEncoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length
        self.in_channel = in_channel
        self.img_size = img_size

        self.block1 = conv_bn_lrelu(self.in_channel, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(128*pow(self.img_size//16, 2), 32)
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, img):
        """
        input is only the image [bs,3,256,256] without concated anatomy factor
        """
        out = self.block1(img)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class UNet(nn.Module):
    def __init__(self, in_channel, width, height, ndf, num_output_channels, normalization, upsample):
        super(UNet, self).__init__()
        """
        UNet autoencoder
        """
        self.in_channel = in_channel
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet(self.in_channel, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.encoder_block3 = conv_block_unet(self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm)
        self.encoder_block4 = conv_block_unet(self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet(self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm)
        self.decoder_upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet(self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm)
        self.decoder_upsample3 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet(self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm)
        self.decoder_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet(self.ndf * 2, self.ndf, 3, 1, 1, self.norm)
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

    def forward(self, x):
        s1 = self.encoder_block1(x)
        out = self.maxpool(s1)
        s2 = self.encoder_block2(out)
        out = self.maxpool(s2)
        s3 = self.encoder_block3(out)
        out = self.maxpool(s3)
        s4 = self.encoder_block4(out)
        out = self.maxpool(s4)

        out = self.bottleneck(out)

        out = self.decoder_upsample1(out)
        out = self.decoder_upconv1(out)
        out = torch.cat((out, s4), 1)
        out = self.decoder_block1(out)
        out = self.decoder_upsample2(out)
        out = self.decoder_upconv2(out)
        out = torch.cat((out, s3), 1)
        out = self.decoder_block2(out)
        out = self.decoder_upsample3(out)
        out = self.decoder_upconv3(out)
        out = torch.cat((out, s2), 1)
        out = self.decoder_block3(out)
        out = self.decoder_upsample4(out)
        out = self.decoder_upconv4(out)
        out = torch.cat((out, s1), 1)
        out = self.decoder_block4(out)
        out = self.classifier_conv(out)

        return out
