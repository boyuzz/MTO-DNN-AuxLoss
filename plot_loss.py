import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import stats
from matplotlib import rc
from matplotlib import rcParams

# Set font type and size
rc('font', family='Times New Roman')
rc('text', usetex=True)
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

loss = ['MSE', 'MAE', 'SSIME']
# loss = ['MSE']


class Result:
	def __init__(self, path, all_run, full_epoch, trans_step, name):
		# comloss = ['l1_mse', 'l1_ssim', 'mse_ssim', 'l1_mse_ssim']
		comloss = ['l1_mse', 'mse_ssim', 'l1_mse_ssim']
		fileloss = ['mse', 'l1', 'ssim']
		# fileloss = ['mse']

		self.multi_mean = []
		self.single_mean = []
		self.com_mean = []
		self.final_loss = np.zeros([all_run, len(fileloss), 2])
		self.com_final_loss = np.zeros([all_run, len(comloss)])
		self.full_epoch = full_epoch
		self.trans_step = trans_step
		self.mx_axis = None
		self.sx_axis = None
		self.name = name
		self.copy_record = []

		for run in range(all_run):
			# res_dir = 'T91_result/frequency_1epoch_random_start/{}/'.format(str(run))
			res_dir = os.path.join(path, str(run))
			# try:
			# multi_model_loss = np.loadtxt(res_dir+'1.0loss.txt')
			multi_model_loss = np.loadtxt(os.path.join(res_dir, '1.0loss.txt'))
			self.multi_mean.append(multi_model_loss[1:, :])
			self.mx_axis = multi_model_loss[0, :]

			single_model_loss = np.loadtxt(os.path.join(res_dir, '0.0loss.txt'))
			self.single_mean.append(single_model_loss[1:, :])
			self.sx_axis = single_model_loss[0, :]

			# com_model_loss = np.loadtxt(os.path.join(res_dir, 'com_loss.txt'))
			# self.com_mean.append(com_model_loss[1:, :])
			# self.cx_axis = com_model_loss[0, :]
			# self.com_final_loss[run-1, :] = np.array([com_model_loss[1:, -1]])

			self.final_loss[run - 1, :, :] = np.array(
				[multi_model_loss[1:, -1], single_model_loss[1:, -1]]).transpose()

			run_copy_record = []
			for l in fileloss:
				filename = os.path.join(res_dir, '1.0' + l + '.txt')
				run_copy_record.append(filename)
			self.copy_record.append(run_copy_record)


			# except:
			# 	print('Load {} failed!'.format(run))
			# 	continue

	def print_mto_combine(self):
		mse_mae = []
		mae_ssim = []
		mse_ssim = []
		mse_mae_ssim = []
		for run in range(all_run):
			mse_mae.append(self.final_loss[run, 0,0] + self.final_loss[run, 1,0])
			mae_ssim.append(self.final_loss[run, 1,0] + self.final_loss[run, 2,0])
			mse_ssim.append(self.final_loss[run, 0,0] + self.final_loss[run, 2,0])
			mse_mae_ssim.append(self.final_loss[run, 0,0] + self.final_loss[run, 1,0] + self.final_loss[run, 2,0])

		mse_mae = np.array(mse_mae)
		mae_ssim = np.array(mae_ssim)
		mse_ssim = np.array(mse_ssim)
		mse_mae_ssim = np.array(mse_mae_ssim)
		print("J1 > MTO MSE+MAE: {}, significant {}".format(self.com_final_loss[:,0].mean()> mse_mae.mean(),
		                                                    stats.wilcoxon(self.com_final_loss[:,0], mse_mae)))
		print("J2 > MTO MAE+SSIME: {}, significant {}".format(self.com_final_loss[:,1].mean()> mae_ssim.mean(),
		                                                      stats.wilcoxon(self.com_final_loss[:, 1], mae_ssim)))
		print("J3 > MTO MSE+SSIME: {}, significant {}".format(self.com_final_loss[:,2].mean()> mse_ssim.mean(),
		                                                      stats.wilcoxon(self.com_final_loss[:, 2], mse_ssim)))
		print("J4 > MTO MSE+MAE+SSIME: {}, significant {}".format(self.com_final_loss[:,3].mean()> mse_mae_ssim.mean(),
		                                                          stats.wilcoxon(self.com_final_loss[:, 3], mse_mae_ssim)))


class Validation:
	def __init__(self, path, all_run):
		comloss = ['l1_mse', 'mse_ssim', 'l1_mse_ssim']
		self.com_final_psnr = np.zeros([all_run, len(comloss)])
		self.com_final_ssim = np.zeros([all_run, len(comloss)])
		dataset = path.split('/')[2]
		model = path.split('/')[-3]

		for run in range(all_run):
			result_file = '{}_test_result_{}_{}'.format(dataset, model, run)
			with open(os.path.join(path, result_file), 'r') as f:
				article = f.readlines()
				pc = 0
				sc = 0
				for line in article:
					if 'psnr' in line.split():
						psnr = float(line.split()[-1])
						self.com_final_psnr[run, pc] = psnr
						pc += 1

					if 'ssim' in line.split():
						ssim = float(line.split()[-1])
						self.com_final_ssim[run, sc] = ssim
						sc += 1

		print('average psnr {} {} results: {}'.format(model, dataset, self.com_final_psnr.mean(0)))
		print('average ssim {} {} results: {}'.format(model, dataset, self.com_final_ssim.mean(0)))


def draw_statistic(objects):
	res_dir = './'
	fig, axes = plt.subplots(len(loss), len(objects), sharex='col', sharey=False, figsize=(13, 8))
	for x, l in enumerate(loss):
		for y, obj in enumerate(objects):
			axes[x,y].boxplot(obj.final_loss[:, x], 0, '', labels=['MTO', 'STO'])
			axes[x,y].yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。

			if y == 0:
				axes[x,y].set_ylabel('Task {} training loss'.format(loss[x]), fontsize=12, rotation='vertical')

			if x == len(loss)-1:
				axes[x,y].set_xlabel(obj.name, fontsize=12)

			for i in range(1, obj.final_loss.shape[2]):
				print('{} MTO mean {}, STO mean {}'.format(l, np.mean(obj.final_loss[:, x, 0]), np.mean(obj.final_loss[:, x, i])))
				print(stats.wilcoxon(obj.final_loss[:, x, 0], obj.final_loss[:, x, i]))

	# plt.ylabel('Training loss function value', fontsize=20)
	plt.tight_layout()
	plt.subplots_adjust(wspace=0.29, hspace=0.15)
	plt.savefig(res_dir + 'box.pdf')


def draw_curve(objects, run):
	res_dir = './'
	fig, axes = plt.subplots(len(loss), len(objects), sharex='col', sharey=False, figsize=(14, 7.5))

	for x, l in enumerate(loss):
		for y, obj in enumerate(objects):
			axes[x, y].set_xlim(-obj.trans_step, obj.full_epoch + obj.trans_step)
			axes[x, y].yaxis.get_major_formatter().set_powerlimits((0, 1))
			axes[x, y].plot(obj.mx_axis.astype(np.int), obj.multi_mean[run][x, :], label='MTO {}'.format(l))
			axes[x, y].plot(obj.sx_axis.astype(np.int), obj.single_mean[run][x, :], '--', color='red',
			                label='STO {}'.format(l))

			filename = obj.copy_record[run][x]
			with open(filename, 'r') as fp:
				content = fp.readlines()
				for idl, line in enumerate(content):
					copied_from = line.strip('\n').split(' ')[-1]
					if copied_from != 'copy' and copied_from != l:
						axes[x, y].scatter(idl * obj.trans_step, obj.multi_mean[run][x, idl],
						                   color='', marker='o', edgecolors='g', s=30)

			if y == 0:
				axes[x, y].set_ylabel('Task {} training loss'.format(loss[x]), fontsize=12, rotation='vertical')

			if x == len(loss) - 1:
				axes[x, y].set_xlabel('Training epoch\n{}'.format(obj.name), fontsize=12)
			axes[x, y].legend(fontsize=10)

	plt.tight_layout()
	plt.subplots_adjust(wspace=0.22, hspace=0.22)
	plt.savefig(res_dir + 'loss_compare.pdf')


if __name__ == '__main__':
	all_run = 15

	# path = './result/ipiu/vdsr8x1e-03/FULLTRAIN_SUPER_R4_RAN0/'
	# ipiuv = Result(path, all_run, full_epoch=100, trans_step=4, name='IPIU-VDSR')
	#
	# path = './result/ipiu/edsr8x1e-03/FULLTRAIN_SUPER_R4_RAN0/'
	# ipiue = Result(path, all_run, full_epoch=100, trans_step=4, name='IPIU-EDSR')
	#
	# path = './result/lfw/vdsr8x1e-04/FULLTRAIN_SUPER_R4_RAN0/'
	# lfwv = Result(path, all_run, full_epoch=100, trans_step=4, name='LFW-VDSR')
	#
	# path = './result/lfw/edsr8x1e-04/FULLTRAIN_SUPER_R4_RAN0/'
	# lfwe = Result(path, all_run, full_epoch=100, trans_step=4, name='LFW-EDSR')
	#
	# path = './result/rsscn7/vdsr4x1e-03/FULLTRAIN_SUPER_R4_RAN0/'
	# rsscn7v = Result(path, all_run, full_epoch=100, trans_step=4, name='RSSCN7-VDSR')
	#
	# path = './result/rsscn7/edsr4x1e-03/FULLTRAIN_SUPER_R4_RAN0/'
	# rsscn7e = Result(path, all_run, full_epoch=100, trans_step=4, name='RSSCN7-EDSR')

	path = './result/ipiu/vdsr8x1e-03/record_mtl/'
	ipiuv = Validation(path, all_run)

	path = './result/ipiu/edsr8x1e-03/record_mtl/'
	ipiue = Validation(path, all_run)

	path = './result/lfw/vdsr8x1e-04/record_mtl/'
	lfwv = Validation(path, all_run)

	path = './result/lfw/edsr8x1e-04/record_mtl/'
	lfwe = Validation(path, all_run)

	path = './result/rsscn7/vdsr4x1e-03/record_mtl/'
	rsscn7v = Validation(path, all_run)

	path = './result/rsscn7/edsr4x1e-03/record_mtl/'
	rsscn7e = Validation(path, all_run)

	objects = [ipiuv, ipiue, lfwv, lfwe, rsscn7v, rsscn7e]
	# objects = [ipiuv, ipiue]
	draw_statistic(objects)
	# draw_curve(objects, run=8)
	# for obj in objects:
	#     obj.print_mto_combine()
	#     print('-'*8)
