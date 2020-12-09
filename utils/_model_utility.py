

import operator
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.style.use('ggplot')




def TrainEvalMLP(model, device, e, train_iter, valid_iter, criterion):
	# =============
	# training loop
	# =============
	running_train_loss = 0.
	for idx, sample in enumerate(train_iter): # len(train_iter) iterations
		images, labels = sample
		images = Variable(images.float().to(device))
		images = images.view(images.size(0), -1) # flatten the image
		output = model(images) # feeding images to the model
		winners = output.argmax(dim=1) # calculate the most prob of predictions, return images.size(0) indices of most prob chars in each row
		labels_long_tensor = labels.nonzero(as_tuple=True) # all nonezero values with their coressponding indices
		corrects = (winners == labels_long_tensor[1]) # list of corrects - labels_long_tensor[0] is batch indices and labels_long_tensor[1] is indices of nonzero values
		train_acc = 100*corrects.sum().float()/float(labels.size(0)) # the ratio of number of correct predictions to the total number of input samples
		train_loss = criterion(output, labels.argmax(dim=1)) # calculate the loss between output and labels
		running_train_loss += train_loss.item()
		model.train(images, labels) # run backpropagation algorithm to tune the weights at the end of the iteration - minibatch gradient
		if idx % 20 == 0: # log every 20 mini-batch
			print("[TRAINING LOG]")
			print('\t☕️ [epoch ⇀  %d, sample/mini-batch ⇀  %d, batch size ⇀  %d] \n\t\t ↳  loss: %.3f - acc: %.3f' % (e + 1, idx + 1, images.size(0), running_train_loss/20, train_acc))
			running_train_loss = 0.
	# ===============
	# validating loop
	# ===============
	running_valid_loss = 0.
	for idx, sample in enumerate(valid_iter): # len(valid_iter) iterations
		images, labels = sample
		images = Variable(images.float().to(device))
		images = images.view(images.size(0), -1) # flatten the image
		output = model(images) # feeding images to the model
		winners = output.argmax(dim=1) # calculate the most prob of predictions, return images.size(0) indices of most prob chars in each row
		labels_long_tensor = labels.nonzero(as_tuple=True) # all nonezero values with their coressponding indices
		corrects = (winners == labels_long_tensor[1]) # list of corrects - labels_long_tensor[0] is batch indices and labels_long_tensor[1] is indices of nonzero values
		valid_acc = 100*corrects.sum().float()/float(labels.size(0)) # the ratio of number of correct predictions to the total number of input samples
		valid_loss = criterion(output, labels.argmax(dim=1)) # calculate the loss between output and labels
		if idx % 20 == 0: # log every 20 mini-batch
			print("\n[VALIDATING LOG]")
			print('\t☕️ [epoch ⇀  %d, sample/mini-batch ⇀  %d, batch size ⇀  %d] \n\t\t ↳  loss: %.3f - acc: %.3f' % (e + 1, idx + 1, images.size(0), running_train_loss/20, train_acc))
			running_train_loss = 0.
	# return the last loss and last acc of both loaders at the end of iteration in an epoch
	return train_loss, train_acc, valid_loss, valid_acc




def TrainEvalCNN(model, device, e, train_iter, valid_iter, optimizer, criterion):
	model.train()
	running_train_loss = 0.
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



def PlotStat(history):
	print(history)