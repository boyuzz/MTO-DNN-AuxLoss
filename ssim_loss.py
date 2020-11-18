import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=False)

    return window


def _ssim(img1, img2, window, window_size, channel, cs_map=False, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    if cs_map:
        cs = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        value = ((2.0 * mu1_mu2 + C1) * cs) / (mu1_sq + mu2_sq + C1)

        if size_average:
            return value.mean(), cs
        else:
            return value.mean(1).mean(1).mean(1).sum(), cs
    else:
        value = ((2.0*mu1_mu2 + C1)*(2.0*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return value.mean()
        else:
            return value.mean(1).mean(1).mean(1).sum()


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (num, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).type_as(img1)
            self.window = window
            self.channel = channel
        if self.size_average:
            return 1-_ssim(img1, img2, window, self.window_size, channel, cs_map=False, size_average=self.size_average)
        else:
            return num-_ssim(img1, img2, window, self.window_size, channel, cs_map=False, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size = 7, size_average = True):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).type_as(img1)
            self.window = window
            self.channel = channel

        weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  #, 0.1333
        mssim = []
        mcs = []
        level = len(weight)
        for l in range(level):
            ssim_map, cs_map = _ssim(img1, img2, window, self.window_size, channel, cs_map=True, size_average=self.size_average)
            mssim.append(torch.mean(ssim_map))
            mcs.append(torch.mean(cs_map))
            img1 = F.avg_pool2d(img1, kernel_size=2)
            img2 = F.avg_pool2d(img2, kernel_size=2)
            # img1 = filtered_im1
            # img2 = filtered_im2

        mssim = (torch.stack(mssim, dim=0)+1)/2
        mcs = (torch.transpose(torch.stack(mcs, dim=0), 0, 1).squeeze(0)+1)/2

        value = (torch.prod(mcs[0:-1] ** weight[0:-1]) *
                 (mssim[-1] ** weight[-1]))

        if self.size_average:
            value = torch.mean(value)

        return 1 - value
