

## Persian Database Classification task using PyTorch

Author: Mohammaderfan Arefimoghaddam([marefimoghaddam@unixerr.com](mailto:marefimoghaddam@unixerr.com))

If you have any question, please feel free to contact me.

[IFHCDB database](http://ele.aut.ac.ir/~imageproc/downloads/ifhcdb.rar)

[IFHCDB paper](https://hal.inria.fr/inria-00112676/document)

To extract the data please contact with [Dr. Karim Faez](mailto:kfaezaut.ac.ir)

## Performance

MLP:

CNN:

## Setup

* Create an environment: ```conda create -n uniXerr```
* Create the environment using the _scai.yml_ file: ```conda env create -f scai.yml```
* Activate _scai_ environment: ```conda activate scai```
* Update the environment using _scai.yml_ file: ```conda env update -f scai.yml --prune```
* Export your active environment to _scai.yml_ file: ```conda env export | grep -v "^prefix: " > scai.yml```

## Usage
```console
python classifier.py --network mlp --batch-size 32 --num-workers 4 --epoch 200 --learning-rate 0.001 --device cpu
```

```console
python classifier.py --pre-trained-model path/to/mlp.pth
```

```console
python classifier.py --pre-trained-model path/to/cnn.pth
```


## Procedures

#### Preprocessing

Both models are trained on CSV files which are the numpy arrays of dataset images and their associated labels of **Persian Database** dataset. If you want to preprocess images of another dataset from scratch just run `_img_to_csv.py` script inside `utils` folder to resize them and store their numpy arrays in to their CSV files.

```console
python utils/_img_to_csv.py --path /path/to/dataset --image-size 64
``` 

> After finishing preprocessing script move all CSV files from `utils` folder to `dataset` folder.

#### calculating std and mean of your dataset

In order to normalize the images of your dataset you have to calculate **mean** and **std** of your data, by using one the methods in `_cal_mean_std.py` script inside `utils` folder you can calculate those parameters and normalize your images to build train and valid dataset pipelines.
More information about [calculating **mean** and **std** in PyTorch](https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2).

> Remember to pass dataloader object into those methods.

```code
mean, std = CalMeanStd0(training_dataloader)
```

or

```code
mean, std = CalMeanStd1(training_dataloader)
```

> `classifier.py` script do this automatically for CSV files dataset

#### building pipelines and dataloaders

The dataset pipelines of training and valid data will normalize all images using calculated **mean** and **std** and convert them into PyTorch tensor. Finally we pass pipelines through dataloader object to prepare them for training and evaluating.

#### training and evaluating on selected model

#### prediction