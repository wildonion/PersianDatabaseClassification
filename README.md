

## Persian Database Classification task using PyTorch

Author: Mohammaderfan Arefimoghaddam([marefimoghaddam@unixerr.com](mailto:marefimoghaddam@unixerr.com))

If you have any question, please feel free to contact me.

[IFHCDB database](http://ele.aut.ac.ir/~imageproc/downloads/ifhcdb.rar)

[IFHCDB paper](https://hal.inria.fr/inria-00112676/document)

To extract the data please contact with [Dr. Karim Faez](mailto:kfaezaut.ac.ir)

## Environment Settings

* PyTorch 1.7
* Python 3.8
* CUDA 10.2

## 📝 Performance

✅ MLP:

✅ CNN:

## ⚙️ Setup

```console
pip install requirements.txt && npm install pm2@latest -g

```
> ⚠️ `uvloop` module is not supported by windows!

Download [**Persian Database** dataset CSV files](https://drive.google.com/file/d/1aeg4D1rLPOZoLUwBWvj6EUiLNu2I3onQ/view?usp=sharing) and extract `images.tar.xz` inside `dataset` folder.


## 💻 Usage

Run `trainer.py` for training selected model:

```console
python trainer.py --network mlp --batch-size 32 --num-workers 4 --epoch 200 --learning-rate 0.001 --device cpu
```

After finishing the training process run `bot.py` 🤖 server for prediction using Telegram-bot APIs.  

```console
pm2 start bot.py
```

## 📋 Procedures

#### 📌 Preprocessing

Both models are trained on CSV files which are the numpy arrays of dataset images and their associated labels of **Persian Database** dataset. If you want to preprocess images of another dataset from scratch just run `_img_to_csv.py` script inside `utils` folder to resize them and store their numpy arrays in to their related CSV files.

```console
python utils/_img_to_csv.py --path /path/to/dataset --image-size 64
```

#### 📌 calculating std and mean of your dataset

In order to normalize the images of your dataset you have to calculate **mean** and **std** of your data. By using one the methods in `_cal_mean_std.py` script inside `utils` folder you can calculate those parameters and normalize(standard scaler) your images to build train and valid dataset pipelines.
More information about [calculating **mean** and **std** in **PyTorch**](https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2).

> Remember to pass dataloader object into those methods.

```python
mean, std = CalMeanStd0(training_dataloader)
```

or

```python
mean, std = CalMeanStd1(training_dataloader)
```

> `trainer.py` script do this automatically for CSV files dataset 🙂

#### 📌 building pipelines and dataloaders

The dataset pipelines of training and valid data will normalize all images using calculated **mean** and **std** and convert them into **PyTorch** tensor. Finally we pass pipelines through dataloader object to prepare them for training and evaluating.

#### 📌 training and evaluating on selected model

I coded backpropagation algorithm from scratch using the chain rule of gradient descent optimization technique for training and tuning the weights of MLP model. You can see it in [`backward`]() function.

For the CNN model I used the built in `backward` method in **PyTorch** of the loss function. It'll automatically calculate the weights using computational graph and update them, so you can access the derivative of each weights' tensor using `.grad` attribute.

> 📊 MLP Plotted history

> 📊 CNN Plotted history

#### 📌 prediction

> Start predicting 🔮 with [pdc bot](http://t.me/pdc_pytorch_bot) 😎✌️