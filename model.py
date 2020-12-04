



import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['NetMLP', 'NetCNN']

class MLP(nn.Module):
	def __init__(self, input_neurons, output_neurons, learning_rate):
		"""
			reference : https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/forwardpropagation_backpropagation_gradientdescent/

			image batch size : (batch, C, H, W) ---flattened---> (batch, C*H*W)
		"""
		super(MLP, self).__init__()
		self.learning_rate = learning_rate

		self.input_to_h1  = nn.Linear(input_neurons, 512) # flattened image to 512 neurons in first hidden layer
		self.h1_to_h2     = nn.Linear(512, 256) # 512 neurons in first hidden to 256 neurons in second hidden layer 
		self.h2_to_output = nn.Linear(256, output_neurons) # 256 neurons in second hidden to "n" neurons in output layer
		self.relu         = nn.ReLU()


	def relu_prime(self, y):
		# y[y<=0] = 0
		# y[y>0] = 1
		# return y
		return torch.where(y <= 0, 0, 1)


	def forward(self, batch):
		self.y1 = self.input_to_h1(batch) # y1 = batch * w1
		self.y2 = F.dropout(self.relu(self.y1), p=0.5, training=self.training) # y2 = relu(y1) - active only on training
		self.y3 = self.h1_to_h2(self.y2) # y3 = y2 * w2
		self.y4 = F.dropout(self.relu(self.y3), p=0.5, training=self.training) # y4 = relu(y3) - active only on training
		self.y5 = self.h2_to_output(self.y4) # y5 = y4 * w3
		return self.y5


	def backward(self, batch, y5, actual):
		"""
								=============================
								  COMPUTING GRADIENT FOR w3
								=============================

			dC/dw3 = dC/dy5 * dy5/dw3  => derivative of loss w.r.t w3

			NOTE : we have to transpose y4 matrix in order to do the multiplication process

		"""

		self.dC_dy5   = y5 - actual # output - actual => derivative of loss w.r.t output
		self.dy5_dw3  = self.y4
		self.dC_w3    = torch.matmul(torch.t(self.dy5_dw3), self.dC_dy5)


		"""
								=============================
								  COMPUTING GRADIENT FOR w2
								=============================

			dC/dw2 = dC/dy5 * dy5/dy4 * dy4/dy3 * dy3/dw2 => derivative of loss w.r.t w2

			NOTE : we have to transpose y2 and w2 matrices in order to do the multiplication process

		"""

		self.dy5_dy4  = self.h2_to_output.weight # w2
		self.dy4_dy3  = self.relu_prime(self.y4) # dy4/dy3 = relu'(y4) because relu(y3) = y4 then relu'(relu(y3)) = relu'(y4) 
		self.dy3_dw2  = self.y2
		self.y5_delta = torch.matmul(self.dC_dy5, torch.t(self.dy5_dy4)) * self.dy4_dy3
		self.dC_dw2   = torch.matmul(self.y5_delta, torch.t(self.dy3_w2))

		
		"""
								=============================
								  COMPUTING GRADIENT FOR w1
								=============================

			dC/dw2 = dC/dy5 * dy5/dy4 * dy4/dy3 * dy3/dy2 * dy2/dy1 * dy1/dw1 => derivative of loss w.r.t w1
			
			NOTE : we have to transpose batch and w1 matrices in order to do the multiplication process

		"""

		self.dy3_dy2  = self.h1_to_h2.weight # w1
		self.dy2_dy1  = self.relu_prime(y2) # dy2/dy1 = relu'(y2) because relu(y1) = y2 then relu'(relu(y1)) = relu'(y2) 
		self.y2_delta = torch.matmul(torch.t(self.dy3_dy2), self.y5_delta) * self.dy2_dy1
		self.dy1_dw1  = batch
		self.dC_w1    = torch.matmul(self.y2_delta, torch.t(self.dy1_dw1))


		"""
									=======================
										UPDATING WEIGHTS
										Δw = α * ∂Eⱼ/∂wᵢ
									=======================

			w1 = w1 - lr*dC/dw1
			w2 = w2 - lr*dC/dw2
			w3 = w3 - lr*dC/dw3

		"""
		
		self.input_to_h1.weight     -= self.learning_rate * self.dC_w1
		self.h1_to_h2.weight        -= self.learning_rate * self.dC_w2
		self.h2_to_output.weight    -= self.learning_rate * self.dC_w3


	def train(self, x, y):
		output = self.forward(x)
		self.backward(x, output, y)




class CNN(nn.Module):
	"""
		image batch size : (batch, C, H, W) 
	"""
	def __init__(self, input_channels, output_neurons):
		super(CNN, self).__init__()
		self.conv1     = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2) 
		self.conv2     = nn.Conv2d(16, 32, kernel_size=5, stride=2)


		"""
		C * H * W is the number of input neurons for fc1 layer which is the flattened batch image
		https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/6
		"""
		input_neurons_for_fc1 = self.conv2.shape[1]*self.conv2.shape[2]*self.conv2.shape[3]


		self.fc1       = nn.Linear(input_neurons_for_fc1, 200)
		self.fc2       = nn.Linear(200, 128)
		self.fc3       = nn.Linear(128, output_neurons)
		self.relu  	   = nn.ReLU()
		self.maxpool2d = nn.MaxPool2d(2, 2)


	def forward(self, batch):
		h1 = self.relu(self.maxpool2d(self.conv1(batch)))
		h1 = F.dropout(h1, p=0.5, training=self.training)

		h2 = self.relu(self.maxpool2d(self.conv2(h1), 2))
		h2 = F.dropout(h2, p=0.5, training=self.training)
		h2 = h2.view(-1, input_neurons_for_fc1)

		h3 = self.relu(self.fc1(h2))
		h3 = F.dropout(h3, p=0.5, training=self.training)

		h4 = self.relu(self.fc2(h3))
		h4 = F.dropout(h4, p=0.5, training=self.training)

		h5 = self.fc2(h4)
		return h4












