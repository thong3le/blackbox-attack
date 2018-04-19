#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python batch_attack.py 0.2 > testing/batch/cifar_un_0.2
CUDA_VISIBLE_DEVICES=0 python batch_attack.py 0.4 > testing/batch/cifar_un_0.4
CUDA_VISIBLE_DEVICES=0 python batch_attack.py 0.8 > testing/batch/cifar_un_0.8
