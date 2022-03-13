import os
import torch
import argparse
import torch.optim as optim
from data import get_training_set
from network import Vgg19, Encoder, Img_decoder_v3, discriminator_v2, Inverse
from libs import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from criterion import LossCriterion_v2, LapLoss, TV, mean_variance_norm
from models import calc_kl, reparameterize
from lpips import lpips


parser = argparse.ArgumentParser()

parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--enc_dir', type=str, default='output/enc_iter_23000_epoch_4.pth')
parser.add_argument('--dec_dir', type=str, default='output/dec_iter_23000_epoch_4.pth')
parser.add_argument('--inv_dir', type=str, default='output/inv_iter_23000_epoch_4.pth')
parser.add_argument("--stylePath", default="/media/server2/HDDShare/sketch",
                    help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="/media/server2/HDDShare/cocoapi-master/coco",
                    help='path to MSCOCO dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
parser.add_argument('--crop', type=bool, default=True, help='crop training images')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument("--outf", default="output/", help='folder to output images and model checkpoints')
parser.add_argument("--content_weight", type=float, default=0.1, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.4, help='style loss weight, 0.02 for origin')
parser.add_argument("--batchSize", type=int, default=4, help='batch size')
parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument("--gpu_id", type=int, default=0, help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')
parser.add_argument("--layer", default="r41", help='which features to transfer, either r31 or r41')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')

################# PREPARATIONS #################
opt = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.makedirs(opt.outf, exist_ok=True)
cudnn.benchmark = True


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


################# DATA #################
print('===> Loading datasets')
train_set = get_training_set(opt.contentPath, opt.stylePath)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

################# MODEL #################
enc = Encoder()
dec = Img_decoder_v3()
inv = Inverse()
VGG = Vgg19()
D = discriminator_v2(device, num_channels=1, base_filter=64)
criterion = LossCriterion_v2(opt.style_weight, opt.content_weight, device=device)

print('---------- encoder architecture -------------')
print_network(enc)



if opt.pretrained:
    if os.path.exists(opt.dec_dir):
        pretrained_dict = torch.load(opt.dec_dir, map_location=lambda storage, loc: storage)
        model_dict = dec.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        dec.load_state_dict(model_dict)
        # dec.load_state_dict(torch.load(opt.dec_dir))
        print('pretrained Decoder model is loaded!')
    if os.path.exists(opt.enc_dir):
        pretrained_dict = torch.load(opt.enc_dir, map_location=lambda storage, loc: storage)
        model_dict = enc.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        enc.load_state_dict(model_dict)
        # dec.load_state_dict(torch.load(opt.dec_dir))
        print('pretrained Encoder model is loaded!')
    if os.path.exists(opt.inv_dir):
        pretrained_dict = torch.load(opt.inv_dir, map_location=lambda storage, loc: storage)
        model_dict = inv.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        inv.load_state_dict(model_dict)
        # dec.load_state_dict(torch.load(opt.dec_dir))
        print('pretrained INverser model is loaded!')

# for param in inv.parameters():
#     param.requires_grad = False
################# LOSS & OPTIMIZER #################
L1_criterion_avg = torch.nn.L1Loss(size_average=True)
BCE_criterion = torch.nn.BCELoss()
Lap_criterion = LapLoss(device=device, max_levels=5)
optimizer_G = optim.AdamW(list(enc.parameters()) + list(dec.parameters()) + list(inv.parameters()), lr=opt.lr)
optimizer_D = optim.AdamW(D.parameters(), lr=opt.lr)


################# GPU  #################
# vgg = torch.nn.DataParallel(vgg)
# dec = torch.nn.DataParallel(dec)
# vgg5 = torch.nn.DataParallel(vgg5)
# matrix = torch.nn.DataParallel(matrix)
# OF_net = torch.nn.DataParallel(OF_net)


D.to(device)
enc.to(device)
dec.to(device)
inv.to(device)
VGG.to(device)
L1_criterion_avg.to(device)
BCE_criterion.to(device)
Lap_criterion.to(device)


################# TRAINING #################

def train(epoch):
    epoch_loss = 0
    enc.train()
    dec.train()
    D.train()
    inv.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        img, edge, edge_gt, ref = batch[0], batch[1], batch[2], batch[3]
        img = img.to(device)
        edge = edge.to(device)
        edge_gt = edge_gt.to(device)
        ref = ref.to(device)

        b, c, h, w = img.shape
        real_label = torch.ones((b, 336)).to(device)
        fake_label = torch.zeros((b, 336)).to(device)

        img_gray = img[:, 0:1, :, :] * 0.299 + img[:, 1:2, :, :] * 0.587 + img[:, 2:3, :, :] * 0.114
        edge_gt = edge_gt[:, 0:1, :, :] * 0.299 + edge_gt[:, 1:2, :, :] * 0.587 + edge_gt[:, 2:3, :, :] * 0.114
        edge = (edge_gt + edge) / 2.0
        # edge = edge_gt * edge
        # edge = edge.clamp(0, 1)
        # img = img.repeat(1, 3, 1, 1)
        # forward
        for param in D.parameters():
            param.requires_grad = False

        optimizer_G.zero_grad()

        feat = enc(img)
        output = dec(feat)
        img_predict = inv(output)
        cF2 = VGG(img_predict)
        cF1 = VGG(img)
        sF = VGG(ref)
        tF = VGG(output.repeat(1, 3, 1, 1))

        # VGG feature
        Laploss = L1_criterion_avg(output, img_gray)
        Imgloss = Lap_criterion(img_predict.repeat(1, 3, 1, 1), img_gray.repeat(1, 3, 1, 1))
        Edgeloss = L1_criterion_avg(output, edge)
        loss_1, styleLoss, contentLoss1= criterion(tF, sF, cF1)
        _, _, contentLoss2 = criterion(cF2, sF, cF1)

        D_fake_feat, D_fake_decision = D(output)

        GAN_loss = L1_criterion_avg(D_fake_decision, real_label)

        G_loss = 5 * GAN_loss + 5 * Laploss + 100 * Imgloss + 100 * Edgeloss + loss_1 + 0.1 * contentLoss2

        # backward & optimization
        G_loss.backward()
        optimizer_G.step()

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        ref = ref[:, 0:1, :, :] * 0.299 + ref[:, 1:2, :, :] * 0.587 + ref[:, 2:3, :, :] * 0.114
        _, D_real_decision = D(ref)
        _, D_fake_decision = D(output.detach())

        real = real_label * np.random.uniform(0.7, 1.2)
        fake = fake_label + np.random.uniform(0.0, 0.3)

        D_loss = (L1_criterion_avg(D_real_decision, real)
                    + L1_criterion_avg(D_fake_decision, fake)) / 2.0

        # Back propagation
        D_loss.backward()
        optimizer_D.step()

        print("===> VAE Epoch[{}]({}/{}): G_loss: {:.4f} || D_loss: {:.4f} || "
              "Imgloss: {:.4f} || Laploss: {:.4f} || Edgeloss: {:.4f} || contentloss1: {:.4f} || "
              "contentloss2: {:.4f} || styleloss: {:.4f} ||"
              "GAN_loss: {:.4f}".format(epoch, iteration,
                                                            len(training_data_loader),
                                                            G_loss.data, D_loss.data,
                                                            Imgloss.data, Laploss.data, Edgeloss.data,
                                                            contentLoss1.data, contentLoss2.data, styleLoss.data,
                                                            GAN_loss.data))


        if iteration % 500 == 0:
            img = img.clamp(0, 1).cpu().data
            edge = edge.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            ref = ref.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            output = output.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            img_predict = img_predict.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            concat = torch.cat((img, ref, edge, output, img_predict), dim=0)
            vutils.save_image(concat, '%s/%d_%d.png' % (opt.outf, epoch, iteration), normalize=True,
                              scale_each=True, nrow=img.shape[0])

            torch.save(enc.state_dict(), 'output/enc_iter_%d_epoch_%d.pth' % (iteration, epoch))
            torch.save(dec.state_dict(), 'output/dec_iter_%d_epoch_%d.pth' % (iteration, epoch))
            torch.save(inv.state_dict(), 'output/inv_iter_%d_epoch_%d.pth' % (iteration, epoch))

    return img, output


for epoch in range(opt.start_iter, opt.nEpochs + 1):
    img, output = train(epoch)
    # content = torch.cat((img1, img2), dim=3)
    # style = style.repeat(1, 1, 1, 2)
    # transfer = torch.cat((transfer1, transfer2), dim=3)
    # learning rate is decayed by a factor of 10 every half of total epochs


