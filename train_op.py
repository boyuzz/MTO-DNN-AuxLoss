from torch.autograd import Variable
import torch.nn as nn
import copy
import util
import torch.optim as optim
import ssim_loss
import numpy as np
from os import listdir
from os.path import join
from architecture.base_networks import *
from PIL import Image


class Train_op():
	def __init__(self, model, optimizer, criterion, name, lr, upscale, cuda, lweights=(1, 1), clip=None, is_feat=False):
		# gpu_id deprecated
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.name = name
		self.lr = lr
		self.upscale = upscale
		self.cuda = cuda
		self.clip = clip
		self.is_feat = is_feat
		self.record = []
		self.lweights = lweights	# only takes effect when using combined loss

		if self.cuda:
			num_gpus = torch.cuda.device_count()
			self.model = nn.DataParallel(self.model, device_ids=list(range(num_gpus))).cuda()
			# self.model = self.model.cuda()
			if isinstance(self.criterion, list):
				self.criterion = [c.cuda() for c in self.criterion]
			else:
				self.criterion = self.criterion.cuda()

	def self_copy(self):
		model = copy.deepcopy(self.model)
		optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
		if len(self.optimizer.state_dict()['state']) != 0:
			optimizer.load_state_dict(self.optimizer.state_dict())
		criterion = self.criterion
		return Train_op(model, optimizer, criterion, self.name, self.lr,
		                self.upscale, self.cuda, self.clip, self.is_feat)

	def evaluate(self, data_loader, ex_criterion=None, feat=False):
		loss = 0
		if ex_criterion is not None:
			criterion = ex_criterion
			is_feat = feat
		else:
			criterion = self.criterion
			is_feat = self.is_feat
		self.model.eval()
		with torch.no_grad():
			for iteration, batch in enumerate(data_loader):
				input, target = batch[0], batch[1]
				if self.cuda:
					target = target.cuda()  # in test loader, pin_memory = True
					input = input.cuda()
				output = self.model(input)
				if is_feat is False:

					if isinstance(criterion, list):
						if 'mtl' not in self.name:
							for c in self.criterion:
								if isinstance(c, nn.L1Loss):
									loss += self.lweights[0] * c(output, target)
								elif isinstance(c, ssim_loss.SSIM):
									loss += self.lweights[1] * c(output, target)
								else:
									loss += c(output, target)
						else:
							loss += criterion[0](output, target).item()
					else:
						loss += criterion(output, target).item()
				else:
					loss += criterion(output, batch).item()

		return loss / len(data_loader)

	def evaluate_lapsrn(self, data_loader, ex_criterion=None, feat=False):

		loss = 0
		iteration = 1

		if ex_criterion is not None:
			criterion = ex_criterion
			is_feat = feat
		else:
			criterion = self.criterion
			is_feat = self.is_feat

		for iteration, batch in enumerate(data_loader, 1):
			input, label_x2, label_x4 = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False), \
			                            Variable(batch[2], requires_grad=False)
			if self.cuda:
				input = input.cuda()
				label_x2 = label_x2.cuda()
				label_x4 = label_x4.cuda()
			if is_feat is False:
				HR_2x, HR_4x = self.model(input)
				loss_x2 = criterion(HR_2x, label_x2).data[0]
				loss_x4 = criterion(HR_4x, label_x4).data[0]
				loss = loss_x2 + loss_x4
			else:
				loss += criterion(self.model(input), batch).data[0]

		return loss / iteration

	def update_lr(self, lr):
		self.lr = lr
		for param_group in self.optimizer.param_groups:
			param_group["lr"] = lr
			# print(param_group["lr"])

	def train_step(self, input_batch, target_batch):
		if self.cuda:
			target_batch = target_batch.cuda()  # in test loader, pin_memory = True
			input_batch = input_batch.cuda()
		self.model.train()
		self.optimizer.zero_grad()
		loss = 0
		output = self.model(input_batch)
		if self.is_feat is False:
			if isinstance(self.criterion, list):
				for idx, c in enumerate(self.criterion):
					if 'mtl' not in self.name:
						if isinstance(c, nn.L1Loss):
							loss += self.lweights[0] * c(output, target_batch)
						elif isinstance(c, ssim_loss.SSIM):
							loss += self.lweights[1] * c(output, target_batch)
						else:
							loss += c(output, target_batch)
					else:
						loss += c(output[idx], target_batch)
			else:
				loss = self.criterion(output, target_batch)
			# loss = self.criterion(output, target_batch)
		else:
			loss = self.criterion(output, batch)

		loss.backward()
		if self.clip is not None:
			nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

		self.optimizer.step()

		return loss.item()

	def train_lap_step(self, input_batch, label_x2, label_x4):

		if self.cuda:
			input_batch = input_batch.cuda()
			label_x2 = label_x2.cuda()
			label_x4 = label_x4.cuda()
		self.model.train()
		self.optimizer.zero_grad()

		HR_2x, HR_4x = self.model(input_batch)
		loss_x2 = self.criterion(HR_2x, label_x2)
		loss_x4 = self.criterion(HR_4x, label_x4)
		loss = loss_x2 + loss_x4
		# else:
		#     loss = self.criterion(output, batch)

		loss_x2.backward(retain_graph=True)
		loss_x4.backward()
		if self.clip is not None:
			nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

		self.optimizer.step()

		return loss.item()

	def add_random(self, ratio):
		param = list(self.model.modules())
		for p in param:
			if isinstance(p, nn.Conv2d):
				shadow = copy.deepcopy(p)
				torch.nn.init.kaiming_normal_(shadow.weight)
				if shadow.bias is not None:
					shadow.bias.data.zero_()
				p.weight.data += ratio*shadow.weight.data

	def copy_parameters(self, _model, alpha=1.0):
		assert alpha >= 0.0 and alpha <= 1.0
		# self.lr = _model.lr
		if alpha == 1.0:
			self.model = copy.deepcopy(_model.model)
			# self.optimizer = copy.deepcopy(_model.optimizer)
			# self.model.load_state_dict(_model.model.module.parameters())
			self.optimizer.load_state_dict(_model.optimizer.state_dict())
		elif alpha == 0:
			return
		else:
			# test_model = copy.deepcopy(model1)
			m1_param = list(self.model.modules())

			m2_param = list(_model.model.modules())

			for i in range(len(m1_param)):
				if isinstance(m1_param[i], nn.Conv2d):
					m1_param[i].weight.data = (1 - alpha) * m1_param[i].weight.data + \
					                          alpha * m2_param[i].weight.data

		if _model.name != self.name:
			self.record.append('copied from {}\n'.format(_model.name))
		else:
			self.record.append('self copy\n')

	def checkpoint(self, epoch, dir, prefix=''):
		model_out_path = dir+prefix+"model_{}_{}_epoch_{}.pth".format(self.name, self.upscale, epoch)
		torch.save(self.model, model_out_path)
		with open('{}.txt'.format(dir+prefix+self.name), 'w') as f:
			for line in self.record:
				f.writelines(line)
		print("Checkpoint saved to {}".format(model_out_path))


