import math
import ssim_loss
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG"])


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def ssim(img1, img2, window_size = 11, size_average = True):
    if isinstance(img1, np.ndarray):
        if len(img1.shape) < 4:
            img1 = img1[np.newaxis, np.newaxis, :]
        img1 = Variable(torch.from_numpy(img1.astype(np.double)), requires_grad=False)
    if isinstance(img2, np.ndarray):
        if len(img2.shape) < 4:
            img2 = img2[np.newaxis, np.newaxis, :]
        img2 = Variable(torch.from_numpy(img2.astype(np.double)), requires_grad=False)

    (_, channel, _, _) = img1.size()
    window = ssim_loss.create_window(window_size, channel).type_as(img1)
    return ssim_loss._ssim(img1, img2, window, window_size, channel, size_average=size_average)


def compare(model_mse, model_ssim):
    loss = 0

    mse_param = list(model_mse.modules())

    ssim_param = list(model_ssim.modules())

    for i in range(len(mse_param)):
        # if isinstance(mse_param[i], vdsr.Conv_ReLU_Block):
        #     residual_mse = list(mse_param[i].modules())
        #     residual_ssim = list(ssim_param[i].modules())
        #     for j in range(len(residual_mse)):
        #         if isinstance(residual_mse[j], nn.Conv2d):
        #             loss += np.mean(np.square(residual_mse[j].weight.data.cpu().numpy() -
        #                                       residual_ssim[j].weight.data.cpu().numpy()))
        if isinstance(mse_param[i], nn.Conv2d):
            loss += np.sqrt(np.square(mse_param[i].weight.data.cpu().numpy() -
                                      ssim_param[i].weight.data.cpu().numpy())).mean()

    print("total difference of two models is ", loss)
    return loss


def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def shave(imgs, border_size=0):
    size = list(imgs.shape)
    if len(size) == 4:
        shave_imgs = torch.FloatTensor(size[0], size[1], size[2]-border_size*2, size[3]-border_size*2)
        for i, img in enumerate(imgs):
            shave_imgs[i, :, :, :] = img[:, border_size:-border_size, border_size:-border_size]
        return shave_imgs
    else:
        return imgs[:, border_size:-border_size, border_size:-border_size]


def variable_shave(imgs, border_size=0):
    # size = list(imgs.data.shape)
    return imgs[:,:, border_size:-border_size, border_size:-border_size]
    # if len(size) == 4:
    #     shave_imgs = torch.FloatTensor(size[0], size[1], size[2]-border_size*2, size[3]-border_size*2)
    #     for i, img in enumerate(imgs):
    #         shave_imgs[i, :, :, :] = img[:, border_size:-border_size, border_size:-border_size]
    #     return shave_imgs


def modcrop(imgs, scale):
    if len(imgs.shape) == 2:
        width, height = imgs.shape
        width = width - np.mod(width, scale)
        height = height - np.mod(height, scale)
        imgs = imgs[:width, :height]
    else:
        width, height, channel = imgs.shape
        width = width - np.mod(width, scale)
        height = height - np.mod(height, scale)
        imgs = imgs[:width, :height, :]

    return imgs


def weights_init_kaming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def preview(training_data_loader):
    dataiter = iter(training_data_loader)
    images, labels = dataiter.next()

    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # show images
    plt.subplot(1, 2, 1)
    imshow(torchvision.utils.make_grid(images))
    plt.subplot(1, 2, 2)
    imshow(torchvision.utils.make_grid(labels))
    plt.show()
    print('===> Building models')