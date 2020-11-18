import argparse
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os
from skimage import io, color, measure
from os.path import join
import util
from os import listdir

from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

dataset = 'ipiu'
form = 'mtl'   # ssim, l1, mse, l1_mse, l1_ssim, mse_ssim, l1_mse_ssim
scale = 8       # upscaling factor
prefix = [form]

#### Specify the model directory: model_dir
model_dir = "./result/{}/edsr{}x1e-03/FULLTRAIN_SUPER_R4_RAN0/4/".format(dataset, scale)
# paths = ["{}0.0model_{}_4_epoch_100.pth".format(model_dir, form), "{}1.0model_{}_4_epoch_100.pth".format(model_dir, form)]
# paths = ["{}com_model_{}_{}_epoch_100.pth".format(model_dir, form, scale)]
paths = ["{}0.0model_{}_{}_epoch_100.pth".format(model_dir, form, scale)]
upscale = False if 'edsr' in model_dir else True

parser = argparse.ArgumentParser(description="PyTorch EDSR Test")
parser.add_argument("--cuda", default=True, type=bool, help="use cuda?")
parser.add_argument("--hr_set", default="../data/{}_paper/".format(dataset), type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=scale, type=int, help="scale factor, Default: 3")

save_dir = './result/{}/{}'.format(dataset, form)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG"])


def colorize(y, cb, cr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y*255
    img[:,:,1] = cb*255
    img[:,:,2] = cr*255
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


def resolve(model, im_input, im_gt_y=None):
    out = model(im_input)
    out = out.cpu()

    im_h_y = out.data[0].numpy().astype(float)
    im_h_y = im_h_y.clip(0, 1)

    if im_gt_y is None:
        return im_h_y
    else:
        psnr_model = measure.compare_psnr(im_gt_y, im_h_y[0, :, :],
                                          data_range=1)  # PSNR(im_gt_y, im_b_y, shave_border=opt.scale)
        ssim_model = measure.compare_ssim(im_gt_y, im_h_y[0, :, :],
                                          data_range=1)  # PSNR(im_gt_y, im_b_y, shave_border=opt.scale)
        return im_h_y, psnr_model, ssim_model


def load_model(path):
    m = torch.load(path, pickle_module=pickle, map_location=lambda storage, loc: storage)
    if cuda:
        if isinstance(m, torch.nn.DataParallel):
            m = m.module.cuda()
        else:
            m = m.cuda()
    return m


opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

hr_list = [join(opt.hr_set, x) for x in sorted(listdir(opt.hr_set)) if is_image_file(x)]  #glob.glob(opt.hr_set+"/*.bmp")

models = [load_model(path) for path in paths]


for id, image in enumerate(hr_list):
    im_gt = io.imread(image)
    img = color.rgb2ycbcr(im_gt) / 255
    img = util.modcrop(img, opt.scale)
    (rows, cols, channel) = img.shape
    im_gt_y, im_bic_cb, im_bic_cr = np.split(img, indices_or_sections=channel, axis=2)
    im_gt_y = im_gt_y.squeeze()
    im_bic_cb = im_bic_cb.squeeze()
    im_bic_cr = im_bic_cr.squeeze()

    im_bic_cb = Image.fromarray(im_bic_cb.squeeze())
    im_bic_cb = im_bic_cb.resize((int(cols // opt.scale), int(rows // opt.scale)), resample=Image.BICUBIC)
    im_bic_cb = im_bic_cb.resize((cols, rows), resample=Image.BICUBIC)
    im_bic_cb = np.asarray(im_bic_cb, dtype=np.float64)

    im_bic_cr = Image.fromarray(im_bic_cr.squeeze())
    im_bic_cr = im_bic_cr.resize((int(cols // opt.scale), int(rows // opt.scale)), resample=Image.BICUBIC)
    im_bic_cr = im_bic_cr.resize((cols, rows), resample=Image.BICUBIC)
    im_bic_cr = np.asarray(im_bic_cr, dtype=np.float64)

    im_lr_y = Image.fromarray(im_gt_y)
    im_lr_y = im_lr_y.resize((int(cols // opt.scale), int(rows // opt.scale)), resample=Image.BICUBIC)
    im_bic_y = im_lr_y.resize((cols, rows), resample=Image.BICUBIC)
    im_bic_y = np.asarray(im_bic_y, dtype=np.float64)

    if not upscale:
        im_lr_y = np.asarray(im_lr_y, dtype=np.float64)
    else:
        im_lr_y = im_bic_y

    # im_input = im_b_y/255.
    im_lr_y = Variable(torch.from_numpy(im_lr_y).float()).view(1, -1, im_lr_y.shape[0], im_lr_y.shape[1])
    if cuda:
        im_lr_y = im_lr_y.cuda()

    gt = Variable(torch.from_numpy(im_gt_y).float()).view(1, -1, im_gt_y.shape[0], im_gt_y.shape[1])
    if cuda:
        gt = gt.cuda()

    psnr_bicubic = measure.compare_psnr(im_gt_y, im_bic_y, data_range=1)
    ssim_bicubic = measure.compare_ssim(im_gt_y, im_bic_y, data_range=1)
    print("BIC PSNR_predicted={}, SSIM_predicted={}".format(psnr_bicubic, ssim_bicubic))

    fname, ext = os.path.basename(image).split('.')
    io.imsave(join(save_dir, fname + '_gt.png'), np.asarray(im_gt) / 255.)
    if not upscale:
        img_bic = colorize(im_bic_y, im_bic_cb, im_bic_cr)
    io.imsave(join(save_dir, fname + '_bic_{:.2f}_{:.4f}.png'.format(psnr_bicubic, ssim_bicubic)),
              np.asarray(img_bic))

    for idx, m in enumerate(models):
        out, psnr_model, ssim_model = resolve(m, im_lr_y, im_gt_y)
        print("{} Model PSNR_predicted={}, SSIM_predicted={}".format(prefix[idx], psnr_model, ssim_model))

        print('saving', os.path.basename(image))
        im_h = colorize(out[0, :, :], im_bic_cb, im_bic_cr)
        io.imsave(join(save_dir, fname+'_{}_{}_{:.2f}_{:.4f}.png'.format(form, prefix[idx], psnr_model, ssim_model)), np.asarray(im_h))
