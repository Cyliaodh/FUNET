import torchvision
import torch
from torch import *
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        mid_ch = out_ch
        self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.batchNorm2d = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.batchNorm2d(
            self.relu(self.conv2(self.pad(self.batchNorm2d(self.relu(self.conv1(self.pad(x))))))))  # add BatchNorm


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class FIUNet(nn.Module):
    def __init__(self, num_class, enc_chs=(2, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64)):
        super(FIUNet, self).__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head1 = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.head2 = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.out = nn.Softmax2d()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out1 = self.head1(out)
        out2 = self.head2(out)
        out1 = F.interpolate(out1, (256, 256))
        out2 = F.interpolate(out2, (256, 256))

        return out1, out2


class EncoderTwoInputs(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024), chs2=(128, 512, 1024, 2048)):
        super().__init__()
        self.chs = chs
        self.enc_blocks = nn.ModuleList([Block(chs2[i], chs2[i + 1]) for i in range(len(chs2) - 1)])
        self.pool = nn.Maxpool2d(2)
        self.help_encoder = Encoder(chs)

    def forward(self, x1, x2):
        help_features = self.help_encoder(x2)
        x = Block(1, 64)(x1)
        ftrs = []
        for i, block in enumerate(self.enc_blocks):
            x = torch.cat([x, help_features[i]], dim=1)
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)

        return ftrs


class LayersFusionUnet(nn.Module):
    def __init__(self, num_class, enc_chs=(1, 64, 128, 256, 512, 1024), enc_chs2=(128, 512, 1024, 2048),
                 dec_chs=(2048, 1024, 512, 128)):
        super().__init__()
        self.encoder = EncoderTwoInputs(enc_chs, enc_chs2)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.out = nn.Softmax2d()

    def forward(self, x1, x2):
        enc_ftrs = self.encoder(x1, x2)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out
