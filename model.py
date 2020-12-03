



import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['NetMLP', 'NetCNN']

class MLP(nn.Module):
	def __init__(self, input_neurons, output_neurons, learning_rate):
		"""
			reference : https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/forwardpropagation_backpropagation_gradientdescent/
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
		self.y1  = self.input_to_h1(batch) # y1 = batch * w1
		self.y2  = F.dropout(self.relu(self.y1), p=0.5) # y2 = relu(y1) 
		self.y3 = self.h1_to_h2(self.y2) # y3 = y2 * w2
		self.y4 = F.dropout(self.relu(self.y3), p=0.5) # y4 = relu(y3)
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
	def __init__(self):
		super(CNN, self).__init__()



	def forward(self, batch):
		pass












