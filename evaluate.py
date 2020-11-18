import argparse
import torch

from torch.utils.data import DataLoader
from dataset import TestDatasetFromFolder
import numpy as np

from skimage import measure
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")


parser = argparse.ArgumentParser(description="PyTorch VDSR Test")
parser.add_argument("--cuda", default=True, type=bool, help="use cuda?")
parser.add_argument("--model", default="vdsr8x1e-03", type=str, help="model path")
parser.add_argument('--prefix', type=str, default='FULLTRAIN_SUPER_R4_RAN0', help='result folder')
parser.add_argument("--test_set", default="ipiu", type=str, help="dataset name, Default: Set5")
parser.add_argument("--upscale_factor", default=8, type=int, help="scale factor, Default: 3")
parser.add_argument("--run", default=0, type=int, help="round, Default: 0")
parser.add_argument("--num_channels", default=1, type=int, help="num of channels, Default: 1")
parser.add_argument("--bic_inp", default=1, type=int, help="whether bic interpolate, Default: 1")
parser.add_argument("--epoch_num", default=100, type=int, help="num of channels, Default: 1")


def load_model(path):
	model = torch.load(path, pickle_module=pickle)
	if opt.cuda:
		if isinstance(model, torch.nn.DataParallel):
			model = model.module.cuda()
		else:
			model = model.cuda()
	return model


def load_sub_result(alpha, run, data_loader, loss, save_dir):

	model_path = save_dir + '{}/{}/{}/{}model_{}_{}_epoch_{}.pth'.format(opt.model, opt.prefix, run, alpha,
																loss, opt.upscale_factor, opt.epoch_num)
	try:
		model = load_model(model_path)
	except OSError:
		print("load model {} error".format(run))
		return

	ave_psnr = 0
	ave_ssim = 0
	model.eval()
	with torch.no_grad():
		for lr, hr, bc in data_loader:
			# input data (low resolution image)
			if opt.num_channels == 1:
				y_ = lr[:, 0].unsqueeze(1)
			else:
				y_ = lr

			if opt.cuda:
				y_ = y_.cuda()

			# prediction
			output = model(y_)
			for i, recon_img in enumerate(output):
				out_img = recon_img.cpu().data
				# calculate psnrs
				if opt.num_channels == 1:
					gt_img = hr[i][0]  # .unsqueeze(0)
					out_img = out_img[0]
				else:
					gt_img = hr[i]
				out_img = out_img.cpu().numpy()
				out_img = np.clip(out_img, 0, 1)
				img_psnr = measure.compare_psnr(out_img, gt_img.cpu().numpy(), data_range=1)
				ave_psnr += img_psnr

				img_ssim = measure.compare_ssim(out_img, gt_img.cpu().numpy(), data_range=1)
				ave_ssim += img_ssim

	ave_psnr /= len(data_loader)
	ave_ssim /= len(data_loader)

	return ave_psnr, ave_ssim


def load_result(alpha, run, data_loader, save_dir):
	model_psnr_path = save_dir+'{}/{}/{}/{}model_psnr_{}.pth'.format(opt.model, opt.prefix, run, alpha, opt.upscale_factor)
	model_ssim_path = save_dir+'{}/{}/{}/{}model_dssim_{}.pth'.format(opt.model, opt.prefix, run, alpha, opt.upscale_factor)
	# model_psnr = torch.load(model_psnr_path, pickle_module=pickle)
	# model_ssim = torch.load(model_ssim_path, pickle_module=pickle)
	try:
		model_psnr = load_model(model_psnr_path)
		model_ssim = load_model(model_ssim_path)
	except OSError:
		print("load model {} error".format(run))
		return

	ave_psnr = 0
	ave_ssim = 0
	ave_bic_psnr = 0
	ave_bic_ssim = 0

	model_psnr.eval()
	model_ssim.eval()
	with torch.no_grad():
		for lr, hr, bc in data_loader:
			# input data (low resolution image)
			if opt.num_channels == 1:
				y_ = lr[:, 0].unsqueeze(1)
			else:
				y_ = lr

			if opt.cuda:
				y_ = y_.cuda()

			# prediction
			out_psnr = model_psnr(y_)
			out_ssim = model_ssim(y_)
			# print('length of out_psnr is {}'.format(len(out_psnr)))
			for i, recon_img in enumerate(out_psnr):
				psnr_img = recon_img.cpu().data
				ssim_img = out_ssim[i].cpu().data
				# calculate psnrs
				if opt.num_channels == 1:
					gt_img = hr[i][0]  # .unsqueeze(0)
					psnr_img = psnr_img[0]
					ssim_img = ssim_img[0]
					bc_img = bc[i][0]
				else:
					gt_img = hr[i]
					bc_img = bc[i]

				psnr_img = psnr_img.cpu().numpy()
				psnr_img = np.clip(psnr_img, 0, 1)
				ssim_img = ssim_img.cpu().numpy()
				ssim_img = np.clip(ssim_img, 0, 1)
				img_psnr = measure.compare_psnr(psnr_img, gt_img.cpu().numpy(), data_range=1)
				ave_psnr += img_psnr

				img_ssim = measure.compare_ssim(ssim_img, gt_img.cpu().numpy(), data_range=1)
				ave_ssim += img_ssim

				bic_psnr = measure.compare_psnr(bc_img.cpu().numpy(), gt_img.cpu().numpy(), data_range=1)
				ave_bic_psnr += bic_psnr

				bic_ssim = measure.compare_ssim(bc_img.cpu().numpy(), gt_img.cpu().numpy(), data_range=1)
				ave_bic_ssim += bic_ssim

	ave_psnr /= len(data_loader)
	ave_ssim /= len(data_loader)
	ave_bic_psnr /= len(data_loader)
	ave_bic_ssim /= len(data_loader)

	return ave_psnr, ave_ssim, ave_bic_psnr, ave_bic_ssim


if __name__ == '__main__':
	opt = parser.parse_args()
	if opt.cuda and not torch.cuda.is_available():
		opt.cuda = False
		print("The GPU is not available in this device, running in CPU mode!")

	val_dir = "./data/{}/val".format(opt.test_set)
	val_set = TestDatasetFromFolder(val_dir, is_gray=True, scale_factor=opt.upscale_factor, bic_inp=opt.bic_inp)
	test_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1,
									  shuffle=False)
	save_dir = './result/{}/'.format(opt.test_set)
	losses = ['mse', 'l1', 'ssim']

	for loss in losses:
		psnr_mto, ssim_mto = load_sub_result(1.0, opt.run, test_data_loader, loss, save_dir)
		psnr_sto, ssim_sto = load_sub_result(0.0, opt.run, test_data_loader, loss, save_dir)
		print('{} psnr avg mto, sto'.format(loss), psnr_mto, psnr_sto)
		print('{} ssim avg mto, sto'.format(loss), ssim_mto, ssim_sto)

	#### Uncomment the following code to show combined loss.
	# combine_loss = ['l1_mse', 'mse_ssim', 'l1_mse_ssim']    #'l1_ssim',
	# for loss in combine_loss:
	# 	psnr_co, ssim_co = load_sub_result('com_', opt.run,  test_data_loader, loss, save_dir)
	# 	print(loss)
	# 	print('{} psnr avg mto, cto'.format(loss), psnr_co)
	# 	print('{} ssim avg mto, cto'.format(loss), ssim_co)

	#### Uncomment the following code to show deep mtl.
	# psnr_mtl, ssim_mtl = load_sub_result(0.0, opt.run, test_data_loader, 'mtl', save_dir)
	# print('mtl')
	# print('mtl psnr avg mto, cto', psnr_mtl)  # psnr_mto,
	# print('mtl ssim avg mto, cto', ssim_mtl)  # ssim_mto,

