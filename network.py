import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchvision.models as models


def exists(val):
    return val is not None

# classes


class Inverse(nn.Module):
    def __init__(self):
        super(Inverse, self).__init__()

        self.E = Encoder()
        self.D = Img_decoder_v3()

    def forward(self, x):
        feat = self.E(x.repeat(1, 3, 1, 1))
        out = self.D(feat)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),
        )

        self.cont_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    def forward(self, HR):
        # LR = F.interpolate(LR, scale_factor=8, mode='nearest')
        feat = self.conv(HR)
        feat1 = F.interpolate(feat, size=(32, 32), mode='bilinear')
        feat1 = feat1.view(HR.shape[0], 512, -1)
        feat1, _ = self.cont_attn(feat1.permute(2, 0, 1), feat1.permute(2, 0, 1), feat1.permute(2, 0, 1))
        feat1 = feat1.view(HR.shape[0], 512, 32, 32)
        feat1 = F.interpolate(feat1, size=(feat.shape[2], feat.shape[3]), mode='bilinear')
        feat = feat + feat1

        return feat




class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=0)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(output_channel)
        self.upsample = upsample
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='bilinear', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(self.reflecPad1(x))
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(self.reflecPad2(x))
        x = self.relu(x)
        x = self.norm(x)

        return x_s + x


class Img_decoder_v3(nn.Module):
    def __init__(self):
        super(Img_decoder_v3, self).__init__()

        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = ResidualBlock(64, 64, upsample=False)
        # self.slice0 = nn.Conv2d(64, 1, 3, 1, 1)
        self.map = nn.Conv2d(64, 5, 3, 1, 1)
        self.confidence = nn.Conv2d(64, 5, 3, 1, 1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, feat):
        # reconstruction
        h = self.slice4(feat)
        h = self.slice3(h)
        h = self.slice2(h)
        h = self.slice1(h)
        score = self.confidence(h)
        score = self.soft(score)
        out = self.map(h) * score
        out = torch.sum(out, dim=1).unsqueeze(1)
        # out = out * self.std.type_as(feat) + self.mean.type_as(feat)

        return out


class Img_decoder_v3_m(nn.Module):
    def __init__(self):
        super(Img_decoder_v3_m, self).__init__()

        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = ResidualBlock(64, 64, upsample=False)
        # self.slice0 = nn.Conv2d(64, 1, 3, 1, 1)
        self.map = nn.Conv2d(64, 5, 3, 1, 1)
        self.confidence = nn.Conv2d(64, 5, 3, 1, 1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, feat, t):
        # reconstruction
        h = self.slice4(feat)
        h = self.slice3(h)
        h = self.slice2(h)
        h = self.slice1(h)
        score = self.confidence(h)
        out = torch.cat((score, self.map(h)), dim=2)
        sorted, indices = torch.sort(out, dim=1)
        result = sorted[:, t:t+1, score.shape[2]:2*score.shape[2], :]

        return result


from collections import namedtuple
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = (x - self.mean.type_as(x)) / self.std.type_as(x)
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h

        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)
        # step 1
        # m, std = calc_mean_std(h_relu1_1)
        # style_1 = torch.cat((m, std), dim=1)
        # # step 2
        # m, std = calc_mean_std(h_relu2_1)
        # style_2 = torch.cat((m, std), dim=1)
        # # step 3
        # m, std = calc_mean_std(h_relu3_1)
        # style_3 = torch.cat((m, std), dim=1)
        # # step 4
        # m, std = calc_mean_std(h_relu4_1)
        # style_4 = torch.cat((m, std), dim=1)
        #
        # code = torch.cat((style_1, style_2, style_3, style_4), dim=1).squeeze(2).squeeze(2)

        return out


class discriminator_v2(nn.Module):
    def __init__(self, device, num_channels, base_filter):
        super(discriminator_v2, self).__init__()
        # self.norm = nn.BatchNorm2d(num_channels*2)
        self.input_conv = nn.Conv2d(num_channels, base_filter, 4, 2, 1)#512*256
        self.norm0 = nn.InstanceNorm2d(base_filter)
        self.conv1 = nn.Conv2d(base_filter, base_filter * 2, 4, 2, 1)
        self.norm1 = nn.InstanceNorm2d(base_filter * 2)
        self.conv2 = nn.Conv2d(base_filter * 2, base_filter * 4, 4, 2, 1)
        self.norm2 = nn.InstanceNorm2d(base_filter * 4)
        self.conv3 = nn.Conv2d(base_filter * 4, base_filter * 8, 4, 2, 1)
        self.norm3 = nn.InstanceNorm2d(base_filter * 8)
        self.cont_attn = nn.MultiheadAttention(embed_dim=base_filter * 8, num_heads=8)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.weight = nn.Conv1d(base_filter * 8, 1, 3, 1, 1)

        self.down = nn.UpsamplingBilinear2d(scale_factor=0.5)

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        feat1 = self.act(self.norm0(self.input_conv(x)))
        feat1 = self.act(self.norm1(self.conv1(feat1)))
        feat2 = self.act(self.norm2(self.conv2(feat1)))
        feat3 = self.act(self.norm3(self.conv3(feat2)))
        feat3 = feat3.view(feat1.shape[0], 512, -1)
        feat3, _ = self.cont_attn(feat3.permute(2, 0, 1), feat3.permute(2, 0, 1), feat3.permute(2, 0, 1))
        feat3 = feat3.view(feat1.shape[0], 512, -1)
        out3 = self.weight(feat3).view(feat1.shape[0], -1)

        feat = torch.cat((feat1.view(feat1.shape[0], -1), feat2.view(feat2.shape[0], -1), feat3.view(feat3.shape[0], -1)), 1)
        return feat, out3

    def forward(self, x):
        # mean = mean.expand_as(x)
        # std = std.expand_as(std)
        # x = torch.cat((x, y), dim=1)
        feat1, prob1 = self.encode(x)
        x = self.down(x)
        feat2, prob2 = self.encode(x)
        x = self.down(x)
        feat3, prob3 = self.encode(x)

        feat_out = torch.cat((feat1, feat2, feat3), 1)
        prob_out = torch.cat((prob1, prob2, prob3), 1)

        return feat_out, prob_out

