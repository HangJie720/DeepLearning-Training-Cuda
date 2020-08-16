# CUDA/Eigen for Deep Leaning Training/Inference with MNIST dataset.
## Main work and contributions
Deep learning models, e.g., state-of-the-art convolutional neural networks (CNNs), have been widely applied into Computer Vision tasks, and training these models based on existing deep learning frameworks(TensorFlow/Torch/Caffe) is also readily available. In order to further explore how the underlying AI framework implements model training based on GPU, It is essential to attempt to train deep learning model based on Cudnn/Cublas/Eigen and other open source libraries to deeply understand the training mechanism of deep learning framework.
In this repository, we try to training fewer classic deep learning models(AlexNet, VGG, ResNet, etc.) based on CUDA.

## Run AlexNet in GPU environment

```bash
$ cd mnist-cudnn-alexnet
$ make
$ ./train -b32 -n2000 -m100 -s -x1000 -y10
```

## Expected output
```bash
== MNIST training with CUDNN ==
[TRAIN]
loading ./dataset/train-images-idx3-ubyte
loaded 60000 items..
.. model Configuration ..
CUDA: alexnet_conv1
CUDA: alexnet_relu1
CUDA: alexnet_pool1
CUDA: alexnet_lrn1
CUDA: alexnet_conv2
CUDA: alexnet_relu2
CUDA: alexnet_pool2
CUDA: alexnet_lrn2
CUDA: alexnet_conv3
CUDA: alexnet_relu3
CUDA: alexnet_conv4
CUDA: alexnet_relu4
CUDA: alexnet_conv5
CUDA: alexnet_relu5
CUDA: alexnet_pool5
CUDA: alexnet_lrn5
CUDA: alexnet_dense1
CUDA: alexnet_relu6
CUDA: alexnet_dropout1
CUDA: alexnet_dense2
CUDA: alexnet_relu7
CUDA: alexnet_dropout2
CUDA: alexnet_dense3
CUDA: alexnet_softmax
.. initialized alexnet_conv1 layer ..
.. initialized alexnet_conv2 layer ..
.. initialized alexnet_conv3 layer ..
.. initialized alexnet_conv4 layer ..
.. initialized alexnet_conv5 layer ..
.. initialized alexnet_dense1 layer ..
.. initialized alexnet_dense2 layer ..
.. initialized alexnet_dense3 layer ..
step:  100, loss: 2.222, accuracy: 15.531%
step:  200, loss: 1.073, accuracy: 54.344%
step:  300, loss: 0.286, accuracy: 80.906%
step:  400, loss: 0.339, accuracy: 86.031%
step:  500, loss: 0.020, accuracy: 86.656%
step:  600, loss: 0.062, accuracy: 87.375%
step:  700, loss: 0.202, accuracy: 87.281%
step:  800, loss: 0.296, accuracy: 86.719%
step:  900, loss: 0.005, accuracy: 86.781%
step: 1000, loss: 0.182, accuracy: 87.969%
step: 1100, loss: 0.003, accuracy: 86.906%
step: 1200, loss: 0.925, accuracy: 86.938%
step: 1300, loss: 0.022, accuracy: 87.531%
step: 1400, loss: 0.001, accuracy: 86.938%
step: 1500, loss: 0.004, accuracy: 86.375%
step: 1600, loss: 0.080, accuracy: 87.438%
step: 1700, loss: 0.366, accuracy: 86.938%
step: 1800, loss: 0.237, accuracy: 87.125%
step: 1900, loss: 0.049, accuracy: 87.938%
step: 2000, loss: 2.577, accuracy: 87.250%
[INFERENCE]
loading ./dataset/t10k-images-idx3-ubyte
loaded 10000 items..
loss: 0.889, accuracy: 84.750%
Done.
```

## Features
* Parameter saving and loading
* Network modification
* Learning rate modificiation
* Dataset shuffling
* Debugging
* how ResNet branches forward/backward
* SGD, Adam, RMSprop optimizer
* Add more layers(e.g., Dropout, LRN, FusedBatchNormalization, Add, Pad.)

## Reference
- [Professional CUDA C Programming](https://www.wiley.com/en-cn/Professional+CUDA+C+Programming-p-9781118739310)
- [Learn CUDA Programming/Chapter10](https://github.com/PacktPublishing/Learn-CUDA-Programming)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Caffe2](https://github.com/Biu-G/libtorch_source/tree/7ee192cfc2b680ee6e06e5910e90e322e9e7c387/caffe2)
- [NVIDIA CUDNN DOCUMENTATION](https://docs.nvidia.com/deeplearning/cudnn/index.html)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [mnist-cudnn](https://github.com/haanjack/mnist-cudnn)
- [MNIST-AlexNet-Using-Tensorflow](https://github.com/qzhao19/MNIST-AlexNet-Using-Tensorflow/blob/master/model.py)
- [CV-Alexnet](https://github.com/ShaoQiBNU/CV-Alexnet)