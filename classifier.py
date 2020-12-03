




# TODO : complete readme




# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/khshim/pytorch_mnist/


import time
import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from dataset import PersianAlphabetDataset
from utils import ToTensor, Normalize, UnNormalize, Img2CSV
from model import CNN, MLP


# ------------ argument options
# ------------------------------
parser = argparse.ArgumentParser(description='Classification of Persian Alphabet')
parser.add_argument('--network', action='store', type=str, help='MLP or CNN')
parser.add_argument('--batch-size', action='store', type=int, help='The number of batch size')
parser.add_argument('--num-workers', action='store', type=int, help='The number of workers for dataloader object')
parser.add_argument('--epoch', action='store', type=int, help='The number of epochs')
parser.add_argument('--learning-rate', action='store', type=float, help='Learning rate value')
parser.add_argument('--image-size', action='store', type=int, help='The width and height size of image')
parser.add_argument('--classes', action='store', type=int, help='The number of total classes for network output')
parser.add_argument('--pre-trained-model', action='store', type=str, help='Path to pre-trained model')
parser.add_argument('--device', action='store', type=str, help='The device to attach the torch to')
args = parser.parse_args()



# ------------ setup device
# ------------------------------
cuda = torch.cuda.is_available() if args.device == 'cuda' else None
device = torch.device("cuda" if cuda else "cpu")
torch.backends.cudnn.benchmark = True



# ------------ helper methods
# ------------------------------
def get_sample(dataloader):
	batch_index = torch.randint(len(dataloader), (1,), device=device)[0]
	for batch_ndx, sample in enumerate(dataloader): # total training data = len(dataloader) * inputs.size(0)
		if batch_ndx == batch_index:
			inputs, labels = sample # sample is a mini-batch (a pack of batch No. data) list with two elements : inputs and labels 
			break
	return inputs, labels


def train(model, optimizer, loader):
	# model.train()
	# output = net(Variable(data.to(device)))
	pass

def evaluate(model, loader):
	pass



# ------------ load pre-trained model
# ---------------------------------------
if args.pre_trained_model and os.path.exists(args.pre_trained_model):
	print(f"\n________found existing pre-trained model, start testing________\n")
	try:
		checkpoint = torch.load(args.pre_trained_model)
		print(f"\t➢   loaded pre-trained model from {args.pre_trained_model}\n")
	except IOError:
		print(f"\t➢   can't load pre-trained model from : {args.pre_trained_model}\n")

	if args.pre_trained_model.split("/")[1][:-3] == "mlp":
		# TODO
		# load mlp model
		# start testing using mlp model
		pass
	elif args.pre_trained_model.split("/")[1][:-3] == "cnn":
		# TODO
		# load cnn model
		# start testing using cnn model
		pass




# ------------ start training and evaluating on training and testing data respectively
# ----------------------------------------------------------------------------------------------
else:
	print(f"\n________found no existing pre-trained model, start training________\n")
	

	# --------------------- building model
	# ---------------------------------------------
	if args.network == "mlp":
		net = MLP(input_neurons=args.image_size**2, output_neurons=args.classes, learning_rate=args.learning_rate)
	elif args.network == "cnn":
		net = CNN()
	else:
		print("[?] Not Supported Network!")
		sys.exit(1)



	# --------------------- preparing dataset
	# ---------------------------------------------
	transform = transforms.Compose([ToTensor(), Normalize(mean=[13.4820], std=[49.6567])]) # normalize image per channel - passing one value for std and mean cause we have one channel
	training_transformed = PersianAlphabetDataset(csv_files=['dataset/train_x.csv', 'dataset/train_y.csv'], transform=transform)
	valid_transformed  = PersianAlphabetDataset(csv_files=['dataset/test_x.csv', 'dataset/test_y.csv'], transform=transform)




	# --------------------- building dataloader objects
	# -----------------------------------------------------
	train_iter = DataLoader(training_transformed, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	valid_iter  = DataLoader(valid_transformed, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

	


	#						-----------------------------------------
	# --------------------- calculating mean and std of training data
	# 						-----------------------------------------
	# 
	# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
	# https://stackoverflow.com/questions/60485362/how-to-normalize-images-in-pytorch
	# ------------------------------------------------------------------------------------
	

							# ========================
							#  SOLUTION 1
							# ========================

	# mean = 0.
	# std = 0.
	# nb_samples = 0.
	# for data in train_iter:
	# 	batch_samples = data[0].size(0)
	# 	data = data[0].view(batch_samples, data[0].size(1), -1)
	# 	mean += data.mean(2).sum(0)
	# 	std += data.std(2).sum(0)
	# 	nb_samples += batch_samples

	# mean /= nb_samples
	# std /= nb_samples

	# print(mean)
	# print(std)
	

							# ========================
							#  SOLUTION 2
							# ========================

	# cnt = 0
	# fst_moment = torch.empty(3)
	# snd_moment = torch.empty(3)

	# for _, sample in enumerate(train_iter):
	# 	images, _ = sample
	# 	b, c, h, w = images.shape
	# 	nb_pixels = b * h * w
	# 	sum_ = torch.sum(images, dim=(0, 2, 3))
	# 	sum_of_square = torch.sum(images**2, dim=(0, 2, 3))
	# 	fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
	# 	snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
	# 	cnt += nb_pixels

	# print("mean", fst_moment)
	# print("std", torch.sqrt(snd_moment - fst_moment ** 2))



	# --------------------- getting a sample
	# ---------------------------------------------
	sample = get_sample(train_iter)
	sample_inputs = sample[0]
	sample_labels = sample[1]


	# --------------------- plottig a sample
	# ---------------------------------------------
	plt.figure()
	plt.imshow(sample_inputs[0].permute(1, 2, 0).numpy())
	plt.show()


	# --------------------- training process
	# ---------------------------------------------
	optimizer = optim.Adam(net.parameters(), args.learning_rate)
	train(net.to(device), optimizer, train_iter) # training model process