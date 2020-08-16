# cuda-for-deep-learning
Transparent CUDNN / CUBLAS usage for the deep learning training using MNIST dataset.

# How to use

### Download MNIST dataset

```bash
$ bash download-mnist-dataset.sh
```

### Run in Nvidia GPU environment

```bash
$ make
$ ./train
```

### Run in BI GPU environment

```bash
$ bash bi_make.sh
$ ./train_bi
```

# Expected output
```bash
== CNN training with CUDNN ==
[TRAIN]
loading ./dataset/train-images-idx3-ubyte
loaded 60000 items..
.. model Configuration ..
CUDA: conv1
CUDA: fbn1
CUDA: relu1
CUDA: pool1
CUDA: conv2
CUDA: fbn2
CUDA: relu2
CUDA: pool2
CUDA: dense1
CUDA: relu3
CUDA: dense2
CUDA: softmax
.. initialized conv1 layer ..
.. initialized fbn1 layer ..
.. initialized conv2 layer ..
.. initialized fbn2 layer ..
.. initialized dense1 layer ..
.. initialized dense2 layer ..
step:  200, loss: 0.095, accuracy: 47.961%
step:  400, loss: 0.092, accuracy: 96.105%
step:  600, loss: 0.167, accuracy: 96.893%
step:  800, loss: 0.058, accuracy: 96.912%
step: 1000, loss: 0.076, accuracy: 96.932%
step: 1200, loss: 0.053, accuracy: 96.934%
step: 1400, loss: 0.149, accuracy: 96.916%
step: 1600, loss: 0.170, accuracy: 96.922%
step: 1800, loss: 0.133, accuracy: 96.914%
step: 2000, loss: 0.118, accuracy: 96.920%
step: 2200, loss: 0.042, accuracy: 96.938%
step: 2400, loss: 0.064, accuracy: 96.922%
[INFERENCE]
loading ./dataset/t10k-images-idx3-ubyte
loaded 10000 items..
loss: 6.451, accuracy: 88.200%
Done.
```

# Features
* Parameter saving and loading
* Network modification
* Learning rate modificiation
* Dataset shuffling
* Testing
* Add more layers

All these features requires re-compilation
