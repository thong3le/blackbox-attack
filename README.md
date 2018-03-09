# blackbox-attack


### About

Implementations of the blackbox attack algorithms in Tensorflow and Pytorch. 

### Model description


#### Tensorflow version: 

This is the CNN model that C&W uses in their paper for MNIST data. (https://arxiv.org/abs/1608.04644)


#### Pytorch version: 

This is a simpler CNN model.

stride = 1, padding = 0

Layer 1: Conv2d 5x5x16, BatchNorm(16), ReLU, Max Pooling 2x2

Layer 2: Conv2d 5x5x32, BatchNorm(32), ReLU, Max Pooling 2x2

Layer 3: FC 10


### Pre-requisites

The following steps should be sufficient to get these attacks up and running on
most Linux-based systems.

```bash
pip install pillow scipy numpy tensorflow keras h5py
conda install pytorch torchvision -c pytorch
```
   
#### To run the Tensorflow version (python3.6):

```bash
python blackbox_attack_tensorflow.py
```

#### To run the Pytorch version (python3.6):

```bash
python blackbox_attack_pytorch.py
```
