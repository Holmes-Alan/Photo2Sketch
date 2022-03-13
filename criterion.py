import torch
import torch.nn as nn
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def mean_variance_norm_loss(feat1, feat2, device):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    size = feat1.size()
    normalized_feat1 = (feat1 - mean.expand(size)) / std.expand(size)
    normalized_feat2 = (feat2 - mean.expand(size)) / std.expand(size)
    return normalized_feat1, normalized_feat2

def TV(x):
    b, c, h_x, w_x = x.shape
    h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv + w_tv

class styleLoss(nn.Module):
    def forward(self,input,target):
        ib,ic,ih,iw = input.size()
        iF = input.view(ib,ic,-1)
        iMean = torch.mean(iF,dim=2)
        iCov = GramMatrix()(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb,tc,-1)
        tMean = torch.mean(tF,dim=2)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(size_average=False)(iMean,tMean) + nn.MSELoss(size_average=False)(iCov,tCov)
        return loss/tb

class styleLoss_v2(nn.Module):
    def forward(self,input,target):
        ib,ic,ih,iw = input.size()
        mean_x, var_x = calc_mean_std(input)
        iCov = GramMatrix()(input)
        mean_y, var_y = calc_mean_std(target)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(size_average=True)(mean_x, mean_y) + nn.MSELoss(size_average=True)(var_x, var_y) + nn.MSELoss(size_average=True)(iCov, tCov)
        return loss

class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)

class LossCriterion(nn.Module):
    def __init__(self,style_layers,content_layers,style_weight,content_weight):
        super(LossCriterion,self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self, tF, sF, cF, KL):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i,cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # style loss
        totalStyleLoss = 0
        for i,layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i,sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight

        # KL loss
        KL = torch.sum(KL)

        # laplacian loss
        # Laploss = Lap_criterion(2*ori_content-1, 2*ori_style-1)

        # total loss
        loss = totalStyleLoss + totalContentLoss + KL

        return loss, totalStyleLoss, totalContentLoss, KL



class LossCriterion_v2(nn.Module):
    def __init__(self, style_weight, content_weight, device):
        super(LossCriterion_v2, self).__init__()

        self.style_weight = style_weight
        self.content_weight = content_weight

        self.L2_loss = nn.MSELoss().to(device)

    def forward(self, tF, sF, cF):
        # content loss
        totalContentLoss = (self.L2_loss(tF.relu4_1, cF.relu4_1) +
                            self.L2_loss(tF.relu3_1, cF.relu3_1) +
                            self.L2_loss(tF.relu2_1, cF.relu2_1) +
                            self.L2_loss(tF.relu1_1, cF.relu1_1)) * self.content_weight

        # style loss
        totalStyleLoss = 0
        # weight_list = [100, 30, 2, 1]
        for ft_x, ft_s in zip(tF, sF):
            mean_x, var_x = calc_mean_std(ft_x)
            mean_style, var_style = calc_mean_std(ft_s)
            # iCov = GramMatrix()(ft_x)
            # tCov = GramMatrix()(ft_s)
            totalStyleLoss = totalStyleLoss + self.L2_loss(mean_x, mean_style)
            totalStyleLoss = totalStyleLoss + self.L2_loss(var_x, var_style)
            # totalStyleLoss = totalStyleLoss + 100*self.L2_loss(iCov, tCov)

        totalStyleLoss = totalStyleLoss * self.style_weight


        # total loss
        loss = totalStyleLoss + totalContentLoss

        return loss, totalStyleLoss, totalContentLoss


class LossCriterion_v3(nn.Module):
    def __init__(self, style_weight, content_weight, device):
        super(LossCriterion_v3, self).__init__()

        self.style_weight = style_weight
        self.content_weight = content_weight

        self.L2_loss = nn.MSELoss().to(device)

    def forward(self, tF, sF, cF, KL):
        # content loss
        totalContentLoss = self.L2_loss(tF['r41'], cF['r41']) * self.content_weight

        # style loss
        totalStyleLoss = 0
        weight_list = [100, 30, 2, 1]
        mean_x, var_x = calc_mean_std(tF['r41'])
        mean_style, var_style = calc_mean_std(sF['r41'])
        totalStyleLoss = totalStyleLoss + weight_list[3] * self.L2_loss(mean_x, mean_style)
        totalStyleLoss = totalStyleLoss + weight_list[3] * self.L2_loss(var_x, var_style)

        mean_x, var_x = calc_mean_std(tF['r31'])
        mean_style, var_style = calc_mean_std(sF['r31'])
        totalStyleLoss = totalStyleLoss + weight_list[2] * self.L2_loss(mean_x, mean_style)
        totalStyleLoss = totalStyleLoss + weight_list[2] * self.L2_loss(var_x, var_style)

        mean_x, var_x = calc_mean_std(tF['r21'])
        mean_style, var_style = calc_mean_std(sF['r21'])
        totalStyleLoss = totalStyleLoss + weight_list[1] * self.L2_loss(mean_x, mean_style)
        totalStyleLoss = totalStyleLoss + weight_list[1] * self.L2_loss(var_x, var_style)

        mean_x, var_x = calc_mean_std(tF['r11'])
        mean_style, var_style = calc_mean_std(sF['r11'])
        totalStyleLoss = totalStyleLoss + weight_list[0] * self.L2_loss(mean_x, mean_style)
        totalStyleLoss = totalStyleLoss + weight_list[0] * self.L2_loss(var_x, var_style)

        totalStyleLoss = totalStyleLoss * self.style_weight

        # KL loss
        KL = torch.mean(KL)

        # total loss
        loss = totalStyleLoss + totalContentLoss + 1*KL

        return loss, totalStyleLoss, totalContentLoss, KL


class LossCriterion_GAN(nn.Module):
    def __init__(self,style_layers,content_layers,style_weight,content_weight):
        super(LossCriterion_GAN,self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self, tF, sF, cF):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i,cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # style loss
        totalStyleLoss = 0
        for i,layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i,sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight

        # laplacian loss
        # Laploss = Lap_criterion(2*ori_content-1, 2*ori_style-1)

        # total loss
        loss = totalStyleLoss + totalContentLoss

        return loss


class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def build_gauss_kernel(cuda, size=5, sigma=1.0, n_channels=1):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])

    kernel = kernel.to(cuda)
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


def down_pyramid(img, max_levels=5):
    current = img
    pyr = []
    pyr.append(img)
    for level in range(max_levels):
        img = fnn.interpolate(img, mode='bilinear', scale_factor=0.5)
        pyr.append(img)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, device, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        self.device = device


    def forward(self, input, target, reduce='mean'):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                cuda=self.device, size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1]
            )
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        if reduce is 'mean':
            L1_loss = torch.nn.L1Loss(size_average=True)
            return sum(L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        else:
            L1_loss = torch.nn.L1Loss(size_average=False)
            return sum(L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


# class LapLoss(nn.Module):
#     def __init__(self, device, max_levels=5, k_size=5, sigma=2.0):
#         super(LapLoss, self).__init__()
#         self.max_levels = max_levels
#         self.k_size = k_size
#         self.sigma = sigma
#         self._gauss_kernel = None
#         self.device = device
#
#
#     def forward(self, input, target, reduce='mean'):
#         if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
#             self._gauss_kernel = build_gauss_kernel(
#                 cuda=self.device, size=self.k_size, sigma=self.sigma,
#                 n_channels=input.shape[1]
#             )
#         pyr_input = down_pyramid(input, self.max_levels)
#         pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
#         if reduce is 'mean':
#             L1_loss = torch.nn.L1Loss(size_average=True)
#             return sum(L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
#         else:
#             L1_loss = torch.nn.L1Loss(size_average=False)
#             return sum(L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class LapMap(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapMap, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)

        return pyr_input