

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

#### calculating std and mean for dataset

#### building dataset and dataloader

#### training and evaluating on selected model

#### prediction