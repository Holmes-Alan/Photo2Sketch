import os
import torch
import argparse
import torch.optim as optim
from datasets import is_image_file
from network import Vgg19, Encoder, Img_decoder_v3, discriminator_v2, Inverse
from libs import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from os.path import *
from os import listdir
import time
from PIL import Image
from img_utils import modcrop, rgb2ycbcr, PSNR, SSIM
import lpips

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--enc_dir', type=str, default='models/enc.pth')
parser.add_argument('--dec_dir', type=str, default='models/dec.pth')
parser.add_argument('--inv_dir', type=str, default='models/inv.pth')
parser.add_argument("--image_dataset", default="data/Test/",
                    help='image dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
parser.add_argument('--crop', type=bool, default=True, help='crop training images')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument("--outf", default="Result/", help='folder to output images and model checkpoints')
parser.add_argument("--batchSize", type=int, default=8, help='batch size')
parser.add_argument("--gpu_id", type=int, default=0, help='which gpu to use')

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




################# MODEL #################
enc = Encoder()
dec = Img_decoder_v3()
inv = Inverse()
VGG = Vgg19()
loss_fn_alex_sp = lpips.LPIPS(net='alex')

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



################# GPU  #################
# vgg = torch.nn.DataParallel(vgg)
# dec = torch.nn.DataParallel(dec)
# vgg5 = torch.nn.DataParallel(vgg5)
# matrix = torch.nn.DataParallel(matrix)
# OF_net = torch.nn.DataParallel(OF_net)

enc.to(device)
dec.to(device)
inv.to(device)
VGG.to(device)
loss_fn_alex_sp.to(device)


################# TRAINING #################

def eval():
    enc.eval()
    dec.eval()

    HR_filename = os.path.join(opt.image_dataset, 'example')
    # LR_filename = os.path.join(opt.image_dataset, 'hazy')
    SR_filename = os.path.join(opt.image_dataset, 'our')
    # SLR_filename = os.path.join(opt.image_dataset, 'SLR')

    gt_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    output_image = [join(SR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    # slr_output_image = [join(SLR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]

    count = 0
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_lpips_predicted = 0.0
    t0 = time.time()
    # ran_patch = torch.randint(896, (2,))
    for i in range(gt_image.__len__()):
        HR = Image.open(gt_image[i]).convert('RGB')
        HR = modcrop(HR, 8)
        with torch.no_grad():
            img = transform(HR).unsqueeze(0).to(device)
            feat = enc(img)
            prediction = dec(feat)
            lpips_sp = loss_fn_alex_sp(prediction, img)
            lpips_sp = lpips_sp.mean()
        torch.cuda.empty_cache()

        # print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))
        prediction = prediction.data[0].cpu().squeeze(0)

        prediction = prediction * 255.0
        prediction = prediction.clamp(0, 255)

        Image.fromarray(np.uint8(prediction)).save(output_image[i])

        GT = np.array(HR).astype(np.float32)
        GT_Y = rgb2ycbcr(GT)

        prediction = np.array(prediction).astype(np.float32)

        psnr_predicted = PSNR(prediction, GT_Y, shave_border=4)
        ssim_predicted = SSIM(prediction, GT_Y, shave_border=4)
        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted
        avg_lpips_predicted += lpips_sp

        count += 1

    t1 = time.time()

    avg_psnr_predicted = avg_psnr_predicted / count
    avg_ssim_predicted = avg_ssim_predicted / count
    avg_lpips_predicted = avg_lpips_predicted / count

    avg_time_predicted = t1 - t0
    print("PSNR_predicted= {:.4f} || "
          "SSIM_predicted= {:.4f} || LPIPS_predicted= {:.4f} || Time= {:.4f} ".format(
        avg_psnr_predicted,
        avg_ssim_predicted,
        avg_lpips_predicted,
        avg_time_predicted))


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)

eval()



