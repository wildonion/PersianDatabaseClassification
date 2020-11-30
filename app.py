








# TODO : implement backward algorithm from scratch using torch

# https://github.com/hunkim/PyTorchZeroToAll/blob/master/03_auto_gradient.py
# https://github.com/hunkim/PyTorchZeroToAll/blob/master/02_manual_gradient.py
# https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/forwardpropagation_backpropagation_gradientdescent/

import torch
import torch.nn as nn


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.input_h1  = nn.Linear(784, 512)
		self.h1_h2     = nn.Linear(512, 256)
		self.h2_output = nn.Linear(256, 32)

	def forward(self):
		pass


	def backward(self):
		pass
