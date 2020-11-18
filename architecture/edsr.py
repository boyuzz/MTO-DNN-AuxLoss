from architecture.base_networks import *
import torch.nn as nn
import util


class Net(torch.nn.Module):
	def __init__(self, num_channels, base_filter, num_residuals, scale=4):
		super(Net, self).__init__()

		self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=None, norm=None)

		resnet_blocks = []
		for _ in range(num_residuals):
			resnet_blocks.append(ResnetBlock(base_filter, norm=None))
		self.residual_layers = nn.Sequential(*resnet_blocks)

		self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None)

		if scale == 4:
			self.upscale4x = nn.Sequential(
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				# UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
			)
		elif scale == 8:
			self.upscale4x = nn.Sequential(
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
			)

		self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

	def weight_init(self, mean=0.0, std=0.02):
		for m in self.modules():
			util.weights_init_normal(m, mean=mean, std=std)

	def forward(self, x):
		out = self.input_conv(x)
		residual = out
		out = self.residual_layers(out)
		out = self.mid_conv(out)
		out = torch.add(out, residual)
		out = self.upscale4x(out)
		out = self.output_conv(out)
		return out


class MTL_Net(torch.nn.Module):
	def __init__(self, num_channels, base_filter, num_residuals, scale=4):
		super(MTL_Net, self).__init__()

		self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=None, norm=None)

		resnet_blocks = []
		for _ in range(num_residuals):
			resnet_blocks.append(ResnetBlock(base_filter, norm=None))
		self.residual_layers = nn.Sequential(*resnet_blocks)

		self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None)

		if scale == 4:
			self.upscale4x = nn.Sequential(
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				# UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
			)
		elif scale == 8:
			self.upscale4x = nn.Sequential(
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
				UpsampleBlock(base_filter, base_filter, scale_factor=2, upsample='ps', activation=None, norm=None),
			)

		self.output_conv_1 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
		self.output_conv_2 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
		self.output_conv_3 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

	def weight_init(self, mean=0.0, std=0.02):
		for m in self.modules():
			util.weights_init_normal(m, mean=mean, std=std)

	def forward(self, x):
		out = self.input_conv(x)
		residual = out
		out = self.residual_layers(out)
		out = self.mid_conv(out)
		out = torch.add(out, residual)
		out = self.upscale4x(out)
		out_1 = self.output_conv_1(out)
		if self.training:
			out_2 = self.output_conv_2(out)
			out_3 = self.output_conv_3(out)
			return out_1, out_2, out_3
		else:
			return out_1
