from __future__ import print_function
import argparse
import os
import numpy as np
import time
import copy
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from train_op import Train_op
import ssim_loss
from dataset import TrainDatasetFromFolder, TestDatasetFromFolder
from architecture import edsr, vdsr
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--train_dir', type=str, required=True, help='training data directory')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--inter_frequency", type=int, default=4, help="Frequency of model communication, Default: n=5")
parser.add_argument("--alpha", type=float, default=1.0, help="percentage of copied model, Default: n=1.0")
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.1')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Learning Rate decay term. Default=0.9')
parser.add_argument('--ranlevel', type=float, default=0, help='Learning Rate decay term. Default=0.9')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=2146, help='random seed to use. Default=123')
parser.add_argument('--run', type=int, default=1, help='random seed to use. Default=123')
parser.add_argument('--save_file', type=str, default='loss.txt', help='random seed to use. Default=123')
parser.add_argument('--model', type=str, default='edsr', help='selected model, options: vdsr, edsr. Default=edsr')
parser.add_argument('--patchsize', type=int, default=10, help='patchsize of input')
parser.add_argument('--resume', type=int, default=0, help='resume training')
opt = parser.parse_args()


def train(training_data_loader, models):
	r'''training code, support training multiple models simultaneously
	:param training_data_loader: data loader for training
	:param models: groups of models
	:param epoch: current epoch, for printing and lr decay
	:return: average losses of current epoch
	'''

	losses = np.zeros(len(models))

	for iteration, batch in enumerate(training_data_loader, 1):

		input_batch = batch[0]
		target_batch = batch[1]
		for idx, model in enumerate(models):
			loss = model.train_step(input_batch, target_batch)
			losses[idx] += loss

	return [loss/float(len(training_data_loader)) for loss in losses]


def interchange(models, data_loader, alpha=1.0):
	updated_loss = []
	for _ in range(opt.inter_frequency):
		updated_loss = train(data_loader, models)

	if alpha == 1.0:
		num = len(models)
		results = np.zeros([num, num])

		for i, model in enumerate(models):
			criterion = model.criterion
			is_feat = False
			if model.is_feat is True:
				is_feat = True
			for j, m in enumerate(models):
				results[i, j] = m.evaluate(data_loader, criterion, is_feat)

		min_idx = np.argmin(results, axis=1)
		print(results, min_idx)
		copied_models = [self_copy(m) for m in models]

		for index, min_id in enumerate(min_idx):
			if min_id != index and results[index,index]>results[index,min_id]:
				models[index].copy_parameters(copied_models[min_id], alpha)
				print("copy", models[min_id].name, "to", models[index].name)
				updated_loss.append(results[index, min_id])
			else:
				updated_loss.append(results[index, index])
	return updated_loss


def interchange_im(models, data_loader, val_loader, alpha=1.0):
	''' models communicate with each other
	:param models: groups of models
	:param data_loader: data loader feed into the model for evaluation
	:return: losses of each model after communication
	'''

	num = len(models)
	updated_loss = [0]*num
	if alpha != 0 and num > 1:
		copied_models = [self_copy(m) for m in models]
		for id, m in enumerate(copied_models):
			print('test model {}'.format(m.name))
			groups = []
			num_gpus = torch.cuda.device_count()
			for idx, mx in enumerate(copied_models):
				if idx != id:
					new_model = self_copy(mx, gpu_id=idx % num_gpus)
					new_model.criterion = m.criterion
					new_model.is_feat = m.is_feat
					groups.append(new_model)
			groups.insert(id, self_copy(m, gpu_id=id % num_gpus))
			for _ in range(opt.inter_frequency):
				train(data_loader, groups)

			losses = []
			for xxx in groups:
				losses.append(xxx.evaluate(val_loader))
			results = np.array(losses)
			min_id = np.argmin(results)
			updated_loss[id] = results[min_id]
			models[id].copy_parameters(groups[min_id], alpha)
			print("copy", groups[min_id].name, "to", models[id].name)
	else:
		for _ in range(opt.inter_frequency):
			train(data_loader, models)
		updated_loss = []
		for xxx in models:
			updated_loss.append(xxx.evaluate(val_loader))
		# print(updated_loss)
	return updated_loss


def self_copy(tp, gpu_id=0):
	model = copy.deepcopy(tp.model)
	optimizer = optim.Adam(model.parameters(), lr=tp.lr)
	if len(tp.optimizer.state_dict()['state']) != 0:
		optimizer.load_state_dict(tp.optimizer.state_dict())
	criterion = tp.criterion
	return Train_op(model, optimizer, criterion, tp.name, tp.lr,
	                tp.upscale, tp.cuda, gpu_id, tp.clip, tp.is_feat)


def load_model(path):
	model = torch.load(path, pickle_module=pickle)
	if opt.cuda:
		if isinstance(model, torch.nn.DataParallel):
			model = model.module.cuda()
		else:
			model = model.cuda()
	return model


def main(run):
	print(opt)
	num_gpus = torch.cuda.device_count()
	print("available gpus are", num_gpus)
	num_cpus = multiprocessing.cpu_count()
	print("available cpus are", num_cpus)

	save_dir = './result/{}/{}{}x{:.0e}/FULLTRAIN_SUPER_R{}_RAN{}/'.format(opt.train_dir, opt.model, opt.upscale_factor,
																		   opt.lr, opt.inter_frequency, opt.ranlevel)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	res_dir = save_dir + str(run) + '/'
	if not os.path.exists(res_dir):
		os.makedirs(res_dir)

	cuda = opt.cuda
	if cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without --cuda")

	torch.manual_seed(opt.seed)
	torch.backends.cudnn.enabled = True
	cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True

	np.random.seed(opt.seed)
	if cuda:
		torch.cuda.manual_seed(opt.seed)

	print('===> Loading datasets')

	train_dir = "./data/{}/".format(opt.train_dir)
	train_set = TrainDatasetFromFolder(train_dir, is_gray=True, random_scale=True, crop_size=opt.upscale_factor*opt.patchsize,
	                                   rotate=True, fliplr=True, fliptb=True, scale_factor=opt.upscale_factor, bic_inp=True if opt.model=='vdsr' else False)
	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
	                                  shuffle=True)

	val_set = TestDatasetFromFolder(train_dir, is_gray=True, scale_factor=opt.upscale_factor, bic_inp=True if opt.model=='vdsr' else False)
	validating_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
	                                    shuffle=False)

	if opt.model == 'vdsr':
		model_mse = vdsr.Net(num_channels=1, base_filter=64, num_residuals=18)
		model_l1 = vdsr.Net(num_channels=1, base_filter=64, num_residuals=18)
		model_ssim = vdsr.Net(num_channels=1, base_filter=64, num_residuals=18)
	else:
		model_mse = edsr.Net(num_channels=1, base_filter=64, num_residuals=18, scale=opt.upscale_factor)
		model_l1 = edsr.Net(num_channels=1, base_filter=64, num_residuals=18, scale=opt.upscale_factor)
		model_ssim = edsr.Net(num_channels=1, base_filter=64, num_residuals=18, scale=opt.upscale_factor)

	if opt.resume:
		model_mse = load_model(res_dir+str(opt.alpha)+"model_{}_{}_epoch_{}.pth".format('mse', opt.upscale_factor, opt.nEpochs))
		model_l1 = load_model(res_dir+str(opt.alpha)+"model_{}_{}_epoch_{}.pth".format('l1', opt.upscale_factor, opt.nEpochs))
		model_ssim = load_model(res_dir+str(opt.alpha)+"model_{}_{}_epoch_{}.pth".format('ssim', opt.upscale_factor, opt.nEpochs))

	print('===> Building criterions')
	criterion_mse = nn.MSELoss()
	criterion_l1 = nn.L1Loss()
	criterion_ssim = ssim_loss.SSIM(size_average=False)

	print('===> Building optimizers')
	optimizer_mse = optim.Adam(model_mse.parameters(), lr=opt.lr)
	optimizer_l1 = optim.Adam(model_l1.parameters(), lr=opt.lr)
	optimizer_ssim = optim.Adam(model_ssim.parameters(), lr=opt.lr)

	m1 = Train_op(model_mse, optimizer_mse, criterion_mse, 'mse', opt.lr, opt.upscale_factor, opt.cuda)
	m2 = Train_op(model_l1, optimizer_l1, criterion_l1, 'l1', opt.lr, opt.upscale_factor, opt.cuda)
	m3 = Train_op(model_ssim, optimizer_ssim, criterion_ssim, 'ssim', opt.lr, opt.upscale_factor, opt.cuda)

	models = [m1, m2, m3]

	x_label = []
	y_label = []

	print('===> start training')

	for epoch in range(0, opt.nEpochs+1, opt.inter_frequency):
		tick_time = time.time()
		print('running epoch {}'.format(epoch))

		for m in models:
			lr = opt.lr * (opt.decay_rate ** (epoch // opt.step))
			m.update_lr(lr)
			print('epoch {}, learning rate is {}'.format(epoch, lr))

		update_loss = interchange_im(models, training_data_loader, validating_data_loader, opt.alpha)

		print('evaluated loss is', update_loss)
		x_label.append(epoch)
		y_label.append(update_loss)

		print('this epoch cost {} seconds.'.format(time.time() - tick_time))

		if epoch % (opt.nEpochs // 10) == 0:
			for m in models:
				m.checkpoint(epoch, res_dir, prefix=str(opt.alpha))

	for m in models:
		m.checkpoint(epoch, res_dir, prefix=str(opt.alpha))

	# save obtained losses
	x_label = np.asarray(x_label)
	y_label = np.asarray(y_label).transpose()
	output = np.insert(y_label, 0, x_label, axis=0)

	if len(models) > 1:
		np.savetxt(res_dir+str(opt.alpha) + opt.save_file, output, fmt='%3.5f')
	else:
		np.savetxt(res_dir + 'loss_'+models[0].name+'.txt', output, fmt='%3.5f')


if __name__ == '__main__':
	# the random seeds used in the paper
	seed = [4423, 612, 7099, 4500, 9641, 5038, 9216, 2479, 6540, 806, 6759, 5225, 8389, 7949, 6337, 6674, 2590, 8275, 1266, 82]
	opt.seed = seed[opt.run-1]
	main(opt.run-1)


