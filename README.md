# blackbox-attack


### About

Implementations of the blackbox attack algorithms in Pytorch

### Model description 

There are two CNN models for MNIST dataset: a simple model and C&W model.

Simple Model for MNIST:

stride = 1, padding = 0

Layer 1: Conv2d 5x5x16, BatchNorm(16), ReLU, Max Pooling 2x2

Layer 2: Conv2d 5x5x32, BatchNorm(32), ReLU, Max Pooling 2x2

Layer 3: FC 10

C&W model for MNIST:
This can be found in C&W paper their paper for MNIST data. (https://arxiv.org/abs/1608.04644)


C&W model for CIFAR10:
This can be found in C&W paper their paper for CIFAR10 data. (https://arxiv.org/abs/1608.04644)

### Pre-requisites

The following steps should be sufficient to get these attacks up and running on
most Linux-based systems.

```bash
conda install pytorch torchvision -c pytorch
```
   
#### To train the model
```bash
python3 models.py
```

#### To run the attack:

```bash
python3 blackbox_attack.py
```
 



