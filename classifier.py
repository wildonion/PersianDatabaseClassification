


# coding: utf-8

'''
	Codded By : 
 █     █░ ██▓ ██▓    ▓█████▄  ▒█████   ███▄    █  ██▓ ▒█████   ███▄    █ 
▓█░ █ ░█░▓██▒▓██▒    ▒██▀ ██▌▒██▒  ██▒ ██ ▀█   █ ▓██▒▒██▒  ██▒ ██ ▀█   █ 
▒█░ █ ░█ ▒██▒▒██░    ░██   █▌▒██░  ██▒▓██  ▀█ ██▒▒██▒▒██░  ██▒▓██  ▀█ ██▒
░█░ █ ░█ ░██░▒██░    ░▓█▄   ▌▒██   ██░▓██▒  ▐▌██▒░██░▒██   ██░▓██▒  ▐▌██▒
░░██▒██▓ ░██░░██████▒░▒████▓ ░ ████▓▒░▒██░   ▓██░░██░░ ████▓▒░▒██░   ▓██
-------------------------------------------------------------------------------------------------
| Feature Selection and Dimensionality Reduction using Genetic Algorithm For Breast Cancer Dataset
|-------------------------------------------------------------------------------------------------
|
| USAGE : _______training_______ 
|			python classifier.py --network mlp --batch-size 32 --num-workers 4 --epoch 200 --learning-rate 0.001 --device cpu
|
|
| USAGE : _______predicting using mlp model_______
|			python classifier.py --pre-trained-model path/to/mlp.pth
|
| 
| USAGE : _______predicting using cnn model_______
| 			python classifier.py --pre-trained-model path/to/cnn.pth
|
|
|
|


'''



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
from utils import ToTensor, Normalize, UnNormalize, CalMeanStd0
from model import CNN, MLP


# ------------ argument options
# ------------------------------
parser = argparse.ArgumentParser(description='Classification of Persian Alphabet')
parser.add_argument('--network', action='store', type=str, help='MLP or CNN')
parser.add_argument('--batch-size', action='store', type=int, help='The number of batch size')
parser.add_argument('--num-workers', action='store', type=int, help='The number of workers for dataloader object')
parser.add_argument('--epoch', action='store', type=int, help='The number of epochs')
parser.add_argument('--learning-rate', action='store', type=float, help='Learning rate value')
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
	model.train()
	for e in range(args.epoch):
		for idx, sample in enumerate(loader):
			images, labels = sample
			images = Variable(images.to(device))
			if model.__class__.__name__ == "MLP":
				pass
			elif model.__class__.__name__ == "CNN":
				pass
			else:
				print("[?] Can't Train Unsupported Network")

def evaluate(model, loader):
	pass






# ------------ load pre-trained model for predicting 
# -----------------------------------------------------
if args.pre_trained_model and os.path.exists(args.pre_trained_model):
	print(f"\n________found existing pre-trained model, start testing________\n")
	try:
		checkpoint = torch.load(args.pre_trained_model)
		print(f"\t➢   loaded pre-trained model from {args.pre_trained_model}\n")
	except IOError:
		print(f"\t➢   can't load pre-trained model from : {args.pre_trained_model}\n")

	if args.pre_trained_model.split("/")[1][:-3] == "mlp":
		# TODO - load mlp model
		# start testing using mlp model
		pass
	elif args.pre_trained_model.split("/")[1][:-3] == "cnn":
		# TODO - load cnn model
		# start testing using cnn model
		pass






# ------------ start training and evaluating on training and testing data respectively
# ----------------------------------------------------------------------------------------------
else:
	print(f"\n________found no existing pre-trained model, start training________\n")




	# 					   ------------------------------------------
	# --------------------- calculating mean and std of training data
	# 					   ------------------------------------------
	# 
	# we do not have to pass the ToTensor an Normalize transforms to 
	# PersianAlphabetDataset cause Dataloader will turn images of dataset
	# into tensor and since we're calculating the mean and std there is no
	# need to pass Normalize in here!
	# -----------------------------------------------------------------------
	cal_mean_std_iter = DataLoader(PersianAlphabetDataset(csv_files=['dataset/train_x.csv', 'dataset/train_y.csv']), batch_size=args.batch_size)
	mean, std = CalMeanStd0(cal_mean_std_iter) # you have to pass a dataloader object





	# 					   ------------------
	# --------------------- building dataset
	# 					   ------------------
	# 
	# normalize image using calculated per channel mean and std
	# passing one value for std and mean cause we have one channel
	# --------------------------------------------------------------------
	transform = transforms.Compose([ToTensor(), Normalize(mean=mean, std=std)])
	training_transformed = PersianAlphabetDataset(csv_files=['dataset/train_x.csv', 'dataset/train_y.csv'], transform=transform)
	valid_transformed  = PersianAlphabetDataset(csv_files=['dataset/test_x.csv', 'dataset/test_y.csv'], transform=transform)





	# --------------------- building dataloader objects
	# -----------------------------------------------------	
	train_iter = DataLoader(training_transformed, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	valid_iter  = DataLoader(valid_transformed, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

	

	

	# --------------------- getting a sample
	# ---------------------------------------------
	mini_batch = get_sample(train_iter)
	mini_batch_inputs = mini_batch[0]
	mini_batch_labels = mini_batch[1]

	
	

	# --------------------- plottig a sample
	# ---------------------------------------------
	plt.figure()
	plt.imshow(mini_batch_inputs[0].permute(1, 2, 0).numpy())
	plt.show()


	


	# --------------------- building model
	# ---------------------------------------------
	if args.network == "mlp":
		net = MLP(input_neurons=mini_batch_inputs.shape[2]**2, output_neurons=mini_batch_labels.shape[1], learning_rate=args.learning_rate)
	elif args.network == "cnn":
		net = CNN(input_channels=mini_batch_inputs.shape[1], output_neurons=mini_batch_labels.shape[1])
	else:
		print("[?] Not Supported Network!")
		sys.exit(1)




	# --------------------- training process
	# ---------------------------------------------
	optimizer = optim.Adam(net.parameters(), args.learning_rate)
	# train(net.to(device), optimizer, train_iter) # training model process
	# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	# https://github.com/khshim/pytorch_mnist/