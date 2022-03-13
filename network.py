import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models import *
from rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding
from criterion import adaptive_instance_normalization


def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v, mask = cls_mask)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_, mask = mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

# main classes

class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_size = 224,
        patch_size = 16,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, mask = None):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        n = hp * wp

        # video to patch embeddings

        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        tokens = self.to_patch_embedding(video)

        # add cls token

        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x =  torch.cat((cls_token, tokens), dim = 1)

        # positional embedding

        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        # calculate masking for uneven number of frames

        frame_mask = None
        cls_attn_mask = None
        if exists(mask):
            mask_with_cls = F.pad(mask, (1, 0), value = True)

            frame_mask = repeat(mask_with_cls, 'b f -> (b h n) () f', n = n, h = self.heads)

            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n = n, h = self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value = True)

        # time and space attention

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        return self.to_out(cls_token)


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

class ResidualBlock_v2(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock_v2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=0)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='bilinear', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(self.reflecPad1(x))
        x = self.relu(x)
        x = self.conv2(self.reflecPad2(x))
        x = self.relu(x)

        return x_s + x

class ResidualBlock_v3(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock_v3, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.upsample = upsample
        # self.norm = nn.InstanceNorm2d(output_channel)
        # self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='bilinear', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.relu(x)
        # x = self.norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.norm(x + x_s)

        return x + x_s

class Img_decoder(nn.Module):
    def __init__(self):
        super(Img_decoder, self).__init__()

        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = nn.Conv2d(64, 3, kernel_size=3, padding=0)
        self.reflecPad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.act = nn.Tanh()
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        h = self.slice4(x)
        h = self.slice3(h)
        h = self.slice2(h)
        h = self.slice1(self.reflecPad(h))

        h = h * self.std.type_as(x) + self.mean.type_as(x)

        return h


class Img_decoder_v2(nn.Module):
    def __init__(self):
        super(Img_decoder_v2, self).__init__()

        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = ResidualBlock(64, 64, upsample=False)
        # self.slice0 = nn.Conv2d(64, 1, 3, 1, 1)
        self.prob = nn.Conv2d(64, 2, 3, 1, 1)
        self.soft = nn.Softmax(dim=1)
        # mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # self.register_buffer('mean', mean)
        # self.register_buffer('std',4 = ResidualBlock(64, 64, upsample=False)
        # self.slice0 = nn.Conv2d(64, 1, 3, 1, 1)
        idx = torch.Tensor([0, 1]).view(1, 2, 1, 1)
        self.register_buffer('idx', idx)

    def forward(self, feat):
        # reconstruction
        h = self.slice4(feat)
        h = self.slice3(h)
        h = self.slice2(h)
        h = self.slice1(h)
        out = self.prob(h)
        out = self.soft(out) * self.idx
        out = torch.sum(out, dim=1).unsqueeze(1)
        # out = out * self.std.type_as(feat) + self.mean.type_as(feat)

        return out


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

