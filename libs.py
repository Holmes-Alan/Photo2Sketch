import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """ avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3 """

        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, matrixSize, 3, 1, 1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize)
        # self.fc = nn.Linear(32*64,256*256)

    def forward(self, x):
        out = self.convs(x)
        # 32x8x8
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        # 32x64
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        # 32x32
        out = out.view(out.size(0), -1)
        return self.fc(out)




class VAE_LT(nn.Module):
    def __init__(self, z_dim):
        super(VAE_LT, self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, mu, logvar):
        mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std).type_as(mu)

        return mu + std

    def forward(self, x):
        # 32x8x8
        b, c, h = x.size()
        x = x.view(b, -1)

        z_q_mu, z_q_logvar = self.encode(x).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu, z_q_logvar)
        out = self.decode(z_q)
        out = out.view(b, c, h)

        KL = torch.mean(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)

        return out, KL


class VAE_LT_test(nn.Module):
    def __init__(self, z_dim):
        super(VAE_LT_test, self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, mu, logvar):
        mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std).type_as(mu)

        return mu + std

    def forward(self, x):
        # 32x8x8
        b, c, h = x.size()
        x = x.view(b, -1)

        z_q_mu, z_q_logvar = self.encode(x).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu, z_q_logvar)
        out = self.decode(z_q)
        out = out.view(b, c, h)

        KL = torch.mean(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)

        return out, KL

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    # nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, mu, logvar):
        # mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std).type_as(mu)

        return mu + std

    def forward(self, x):
        # 32x8x8
        b, c, h = x.size()
        x = x.view(b, -1)

        z_q_mu, z_q_logvar = self.encode(x).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu, z_q_logvar)
        out = self.decode(z_q)
        out = out.view(b, c, h)

        KL = torch.mean(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)

        return out, KL


class VAE_test(nn.Module):
    def __init__(self, z_dim):
        super(VAE_test, self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    # nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, mu_x1, logvar_x1, mu_x2, logvar_x2, mu_x3, logvar_x3, mu_x4, logvar_x4, k, l, m, n):
        # mu = self.bn(mu)
        std_x1 = torch.exp(logvar_x1)
        std_x2 = torch.exp(logvar_x2)
        std_x3 = torch.exp(logvar_x3)
        std_x4 = torch.exp(logvar_x4)
        std = torch.pow(k * torch.pow(std_x1, 2) + l * torch.pow(std_x2, 2) + m * torch.pow(std_x3, 2) + n * torch.pow(std_x4, 2), 0.5)
        mu = k * mu_x1 + l * mu_x2 + m * mu_x3 + n * mu_x4
        eps = torch.randn_like(std).type_as(mu_x1)

        return mu + std

    def forward(self, x1, x2, x3, x4, k, l, m, n):
        # 32x8x8
        b, c, h = x1.size()
        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)
        x3 = x3.view(b, -1)
        x4 = x4.view(b, -1)
        z_q_mu_x1, z_q_logvar_x1 = self.encode(x1).chunk(2, dim=1)
        z_q_mu_x2, z_q_logvar_x2 = self.encode(x2).chunk(2, dim=1)
        z_q_mu_x3, z_q_logvar_x3 = self.encode(x3).chunk(2, dim=1)
        z_q_mu_x4, z_q_logvar_x4 = self.encode(x4).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu_x1, z_q_logvar_x1, z_q_mu_x2, z_q_logvar_x2, z_q_mu_x3, z_q_logvar_x3, z_q_mu_x4, z_q_logvar_x4, k, l, m, n)
        out = self.decode(z_q)
        out = out.view(b, c, h)

        return out

class VAE_eval(nn.Module):
    def __init__(self, z_dim):
        super(VAE_eval, self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, k, x_mu, x_logvar, y_mu, y_logvar):
        x_mu = self.bn(x_mu)
        x_std = torch.exp(x_logvar)
        y_mu = self.bn(y_mu)
        y_std = torch.exp(y_logvar)

        mu = (1 - k) * x_mu + k * y_mu
        x_var = torch.pow(x_std, 2)
        y_var = torch.pow(y_std, 2)

        std = torch.pow((1 - k) * x_var + k * y_var, 0.5)

        return mu + std

    def forward(self, k, x, y):
        # 32x8x8
        b, c, h = x.size()
        x = x.view(b, -1)
        y = y.view(b, -1)

        x_mu, x_logvar = self.encode(x).chunk(2, dim=1)
        y_mu, y_logvar = self.encode(y).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(k, x_mu, x_logvar, y_mu, y_logvar)
        out = self.decode(z_q)
        out = out.view(b, c, h)

        return out


class VAE_4x(nn.Module):
    def __init__(self, z_dim):
        super(VAE_4x, self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, k, l, m, n, a_mu, a_logvar, b_mu, b_logvar, c_mu, c_logvar, d_mu, d_logvar):
        a_mu = self.bn(a_mu)
        a_std = torch.exp(a_logvar)
        b_mu = self.bn(b_mu)
        b_std = torch.exp(b_logvar)
        c_mu = self.bn(c_mu)
        c_std = torch.exp(c_logvar)
        d_mu = self.bn(d_mu)
        d_std = torch.exp(d_logvar)

        mu = k * a_mu + l * b_mu + m * c_mu + n * d_mu
        a_var = torch.pow(a_std, 2)
        b_var = torch.pow(b_std, 2)
        c_var = torch.pow(c_std, 2)
        d_var = torch.pow(d_std, 2)

        std = torch.pow(k * a_var + l * b_var + m * c_var + n * d_var, 0.5)

        return mu + std

    def forward(self, k, l, m, n, a, b, c, d):
        # 32x8x8
        batch, cl, h = a.size()
        a = a.view(batch, -1)
        b = b.view(batch, -1)
        c = c.view(batch, -1)
        d = d.view(batch, -1)

        x = torch.cat((a, b, c, d), dim=0)
        mu, logvar = self.encode(x).chunk(2, dim=1)
        a_mu, b_mu, c_mu, d_mu = mu.chunk(4, dim=0)
        a_logvar, b_logvar, c_logvar, d_logvar = logvar.chunk(4, dim=0)
        # reparameterize
        z_q = self.reparameterize(k, l, m, n, a_mu, a_logvar, b_mu, b_logvar, c_mu, c_logvar, d_mu, d_logvar)
        out = self.decode(z_q)
        out = out.view(batch, cl, h)

        return out


class CNN_VAE_eval(nn.Module):
    def __init__(self, layer, z_dim, matrixSize=32):
        super(CNN_VAE_eval, self).__init__()
        if (layer == 'r31'):
            # 256x64x64
            self.convs = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 64, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, matrixSize, 3, 1, 1))
        elif (layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, matrixSize, 3, 1, 1))

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(matrixSize * matrixSize, 2 * z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, matrixSize * matrixSize),
                                    nn.BatchNorm1d(matrixSize * matrixSize),
                                    nn.ReLU(),
                                    nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize),
                                    )

    def reparameterize(self, k, x_mu, x_logvar, y_mu, y_logvar):
        x_mu = self.bn(x_mu)
        x_std = torch.exp(x_logvar)
        y_mu = self.bn(y_mu)
        y_std = torch.exp(y_logvar)

        mu = k * x_mu + (1 - k) * y_mu
        x_var = torch.pow(x_std, 2)
        y_var = torch.pow(y_std, 2)

        std = torch.pow(k * x_var + (1 - k) * y_var, 0.5)

        return mu + std

    def forward(self, k, x, y):
        out_1 = self.convs(x)
        out_2 = self.convs(y)
        # 32x8x8
        b, c, h, w = out_1.size()
        out_1 = out_1.view(b, c, -1)
        out_2 = out_2.view(b, c, -1)
        # 32x64
        out_1 = torch.bmm(out_1, out_1.transpose(1, 2)).div(h * w)
        out_2 = torch.bmm(out_2, out_2.transpose(1, 2)).div(h * w)
        # 32x32
        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = out_2.view(out_2.size(0), -1)

        z_q_mu_1, z_q_logvar_1 = self.encode(out_1).chunk(2, dim=1)
        z_q_mu_2, z_q_logvar_2 = self.encode(out_2).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(k, z_q_mu_1, z_q_logvar_1, z_q_mu_2, z_q_logvar_2)
        out = self.decode(z_q)

        return out



class MulLayer(nn.Module):
    def __init__(self, matrixSize=32):
        super(MulLayer, self).__init__()
        # self.snet = CNN_VAE(layer, z_dim, matrixSize)
        self.snet = CNN(matrixSize)
        self.cnet = CNN(matrixSize)
        # self.VAE = VAE_LT(z_dim=z_dim)
        self.matrixSize = matrixSize

        self.norm = InstanceNorm()
        self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
        self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        # self.reflecPad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.smooth = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, 3, 1, 0, bias=False),
            nn.ReLU()
        )

        # self.act = nn.Tanh()


    def forward(self, cF, sF):
        cF = self.norm(cF)

        sMean, sVar = calc_mean_std(sF)
        # sMean = sMean.squeeze(-1)
        # sMean, KL = self.VAE(sMean)
        # sMean = sMean.unsqueeze(-1)
        sF = (sF - sMean.expand_as(sF)) / sVar.expand_as(sF)

        # out_adain = cF * sVar.expand_as(cF)

        compress_content = self.compress(cF)
        b, c, h, w = compress_content.size()
        compress_content = compress_content.view(b, c, -1)

        cMatrix = self.cnet(cF)
        sMatrix = self.snet(sF)

        sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize)
        cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)
        transmatrix = torch.bmm(sMatrix, cMatrix)
        transfeature = torch.bmm(transmatrix, compress_content).view(b, c, h, w)

        out = self.unzip(transfeature.view(b, c, h, w)) + self.smooth(cF)
        out = self.norm(out) * sVar.expand_as(cF) + sMean.expand_as(cF)


        return out




def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=False):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


# def warp(x, flo, padding_mode='border'):
#     B, C, H, W = x.size()
#
#     # Mesh grid
#     xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
#     yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
#     xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     grid = torch.cat((xx, yy), 1).float()
#     if x.is_cuda:
#         grid = grid.type_as(x)
#     vgrid = grid - flo
#
#     # Scale grid to [-1,1]
#     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
#     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
#     vgrid = vgrid.permute(0, 2, 3, 1)
#     output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='nearest')
#     return output

def warp(x, flo, device):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(device)
    vgrid = grid + flo
    vgrid = vgrid.to(device)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask