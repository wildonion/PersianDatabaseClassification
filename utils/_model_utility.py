

import operator
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


# https://github.com/wildonion/uniXerr/blob/master/core/position_classification/model.py
# https://github.com/khshim/pytorch_mnist/blob/master/main.py


def TrainEvalMLP(model, loader):
	pass
	# loss_lst = []
	# for idx, sample in enumerate(loader):
	# 	images, labels = sample
	# 	images = Variable(images.to(device))
	#     output = model(images)
	#     cross_entropy_loss = -(labels * torch.log(output) + (1 - labels) * torch.log(1 - output))
	#     mean_cross_entropy_loss = torch.mean(cross_entropy_loss).detach().item()
	#     if epoch % 20 == 0:
	#         print('Epoch {} | Loss: {}'.format(epoch, mean_cross_entropy_loss))
	#     loss_lst.append(mean_cross_entropy_loss)
	#     model.train(images, labels)



def TrainEvalCNN(model, loader, optimizer):
	pass
	# criterion = torch.nn.CrossEntropyLoss()
	# model.train()
	# loss_sum = 0
	# acc_sum = 0
 #    for idx, sample in enumerate(loader):
	# 	images, labels = sample
	# 	images = Variable(images.to(device))
 #        optimizer.zero_grad()
	# 	output = model(images)
 #        loss = criterion(output, target)
 #        loss_sum += loss.item()
 #        loss.backward()
 #        optimizer.step()

 #        predict = output.data.max(1)[1]
 #        acc = predict.eq(target.data).cpu().sum()
 #        acc_sum += acc

 #    loss_train = loss_sum / len(loader)
 #    acc_train  =  acc_sum / len(loader) 

	# model.eval()
 #    loss_sum = 0
 #    acc_sum = 0
 #    for idx, (data, target) in enumerate(loader):
 #        data, target = data.cuda(), target.cuda()
 #        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
 #        output = model(data)
 #        loss = F.cross_entropy(output, target)
 #        loss_sum += loss.data[0]

 #        predict = output.data.max(1)[1]
 #        acc = predict.eq(target.data).cpu().sum()
 #        acc_sum += acc

 #    loss_val = loss_sum / len(loader)
 #    acc_val  =  acc_sum / len(loader)

 #    return loss_train, acc_train, loss_val, acc_val



def PlotStat():
	pass