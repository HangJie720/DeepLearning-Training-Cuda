# CUDA/Eigen for Deep Leaning Training/Inference with MNIST dataset.
## Main work and contributions
Deep learning models, e.g., state-of-the-art convolutional neural networks (CNNs), have been widely applied into Computer Vision tasks, and training these models based on existing deep learning frameworks(TensorFlow/Torch/Caffe) is also readily available. In order to further explore how the underlying AI framework implements model training based on GPU, It is essential to attempt to train deep learning model based on Cudnn/Cublas/Eigen and other open source libraries to deeply understand the training mechanism of deep learning framework.
In this repository, we try to training fewer classic deep learning models(AlexNet, VGG, ResNet, etc.) based on CUDA.

## Run resnet50 in GPU environment

```bash
$ cd mnist-cudnn-resnet
$ make
$ ./train -b32 -n500 -m100 -r50
```

## Expected output
```bash
== MNIST training with CUDNN ==
[TRAIN]
loading ./dataset/train-images-idx3-ubyte
loaded 60000 items..
.. model Configuration ..
CUDA: conv2d
CUDA: fbn
CUDA: relu
CUDA: pool
CUDA: conv2d_1
CUDA: fbn_1
CUDA: conv2d_2
CUDA: fbn_2
CUDA: relu_1
CUDA: conv2d_3
CUDA: fbn_3
CUDA: relu_2
CUDA: conv2d_4
CUDA: fbn_4
CUDA: add_1
CUDA: relu_3
CUDA: conv2d_5
CUDA: fbn_5
CUDA: relu_4
CUDA: conv2d_6
CUDA: fbn_6
CUDA: relu_5
CUDA: conv2d_7
CUDA: fbn_7
CUDA: add_2
CUDA: relu_6
CUDA: conv2d_8
CUDA: fbn_8
CUDA: relu_7
CUDA: conv2d_9
CUDA: fbn_9
CUDA: relu_8
CUDA: conv2d_10
CUDA: fbn_10
CUDA: add_3
CUDA: relu_9
CUDA: conv2d_11
CUDA: fbn_11
CUDA: conv2d_12
CUDA: fbn_12
CUDA: relu_10
CUDA: pad_
CUDA: conv2d_13
CUDA: fbn_13
CUDA: relu_11
CUDA: conv2d_14
CUDA: fbn_14
CUDA: add_4
CUDA: relu_12
CUDA: conv2d_15
CUDA: fbn_15
CUDA: relu_13
CUDA: conv2d_16
CUDA: fbn_16
CUDA: relu_14
CUDA: conv2d_17
CUDA: fbn_17
CUDA: add_5
CUDA: relu_15
CUDA: conv2d_18
CUDA: fbn_18
CUDA: relu_16
CUDA: conv2d_19
CUDA: fbn_19
CUDA: relu_17
CUDA: conv2d_20
CUDA: fbn_20
CUDA: add_6
CUDA: relu_18
CUDA: conv2d_21
CUDA: fbn_21
CUDA: relu_19
CUDA: conv2d_22
CUDA: fbn_22
CUDA: relu_20
CUDA: conv2d_23
CUDA: fbn_23
CUDA: add_7
CUDA: relu_21
CUDA: conv2d_24
CUDA: fbn_24
CUDA: relu_22
CUDA: conv2d_25
CUDA: fbn_25
CUDA: relu_23
CUDA: conv2d_26
CUDA: fbn_26
CUDA: add_8
CUDA: relu_24
CUDA: conv2d_27
CUDA: fbn_27
CUDA: relu_25
CUDA: conv2d_28
CUDA: fbn_28
CUDA: relu_26
CUDA: conv2d_29
CUDA: fbn_29
CUDA: add_9
CUDA: relu_27
CUDA: conv2d_30
CUDA: fbn_30
CUDA: conv2d_31
CUDA: fbn_31
CUDA: relu_28
CUDA: pad_
CUDA: conv2d_32
CUDA: fbn_32
CUDA: relu_29
CUDA: conv2d_33
CUDA: fbn_33
CUDA: add_10
CUDA: relu_30
CUDA: conv2d_34
CUDA: fbn_34
CUDA: relu_31
CUDA: conv2d_35
CUDA: fbn_35
CUDA: relu_32
CUDA: conv2d_36
CUDA: fbn_36
CUDA: add_11
CUDA: relu_33
CUDA: conv2d_37
CUDA: fbn_37
CUDA: relu_34
CUDA: conv2d_38
CUDA: fbn_38
CUDA: relu_35
CUDA: conv2d_39
CUDA: fbn_39
CUDA: add_12
CUDA: relu_36
CUDA: conv2d_40
CUDA: fbn_40
CUDA: relu_37
CUDA: conv2d_41
CUDA: fbn_41
CUDA: relu_38
CUDA: conv2d_42
CUDA: fbn_42
CUDA: add_13
CUDA: relu_39
CUDA: conv2d_43
CUDA: fbn_43
CUDA: conv2d_44
CUDA: fbn_44
CUDA: relu_40
CUDA: pad_
CUDA: conv2d_45
CUDA: fbn_45
CUDA: relu_41
CUDA: conv2d_46
CUDA: fbn_46
CUDA: add_14
CUDA: relu_42
CUDA: conv2d_47
CUDA: fbn_47
CUDA: relu_43
CUDA: conv2d_48
CUDA: fbn_48
CUDA: relu_44
CUDA: conv2d_49
CUDA: fbn_49
CUDA: add_15
CUDA: relu_45
CUDA: conv2d_50
CUDA: fbn_50
CUDA: relu_46
CUDA: conv2d_51
CUDA: fbn_51
CUDA: relu_47
CUDA: conv2d_52
CUDA: fbn_52
CUDA: add_16
CUDA: relu_48
CUDA: dense
CUDA: softmax
.. initialized conv2d layer ..
.. initialized fbn layer ..
.. initialized conv2d_1 layer ..
.. initialized fbn_1 layer ..
.. initialized conv2d_2 layer ..
.. initialized fbn_2 layer ..
.. initialized conv2d_3 layer ..
.. initialized fbn_3 layer ..
.. initialized conv2d_4 layer ..
.. initialized fbn_4 layer ..
.. initialized conv2d_5 layer ..
.. initialized fbn_5 layer ..
.. initialized conv2d_6 layer ..
.. initialized fbn_6 layer ..
.. initialized conv2d_7 layer ..
.. initialized fbn_7 layer ..
.. initialized conv2d_8 layer ..
.. initialized fbn_8 layer ..
.. initialized conv2d_9 layer ..
.. initialized fbn_9 layer ..
.. initialized conv2d_10 layer ..
.. initialized fbn_10 layer ..
.. initialized conv2d_11 layer ..
.. initialized fbn_11 layer ..
.. initialized conv2d_12 layer ..
.. initialized fbn_12 layer ..
.. initialized conv2d_13 layer ..
.. initialized fbn_13 layer ..
.. initialized conv2d_14 layer ..
.. initialized fbn_14 layer ..
.. initialized conv2d_15 layer ..
.. initialized fbn_15 layer ..
.. initialized conv2d_16 layer ..
.. initialized fbn_16 layer ..
.. initialized conv2d_17 layer ..
.. initialized fbn_17 layer ..
.. initialized conv2d_18 layer ..
.. initialized fbn_18 layer ..
.. initialized conv2d_19 layer ..
.. initialized fbn_19 layer ..
.. initialized conv2d_20 layer ..
.. initialized fbn_20 layer ..
.. initialized conv2d_21 layer ..
.. initialized fbn_21 layer ..
.. initialized conv2d_22 layer ..
.. initialized fbn_22 layer ..
.. initialized conv2d_23 layer ..
.. initialized fbn_23 layer ..
.. initialized conv2d_24 layer ..
.. initialized fbn_24 layer ..
.. initialized conv2d_25 layer ..
.. initialized fbn_25 layer ..
.. initialized conv2d_26 layer ..
.. initialized fbn_26 layer ..
.. initialized conv2d_27 layer ..
.. initialized fbn_27 layer ..
.. initialized conv2d_28 layer ..
.. initialized fbn_28 layer ..
.. initialized conv2d_29 layer ..
.. initialized fbn_29 layer ..
.. initialized conv2d_30 layer ..
.. initialized fbn_30 layer ..
.. initialized conv2d_31 layer ..
.. initialized fbn_31 layer ..
.. initialized conv2d_32 layer ..
.. initialized fbn_32 layer ..
.. initialized conv2d_33 layer ..
.. initialized fbn_33 layer ..
.. initialized conv2d_34 layer ..
.. initialized fbn_34 layer ..
.. initialized conv2d_35 layer ..
.. initialized fbn_35 layer ..
.. initialized conv2d_36 layer ..
.. initialized fbn_36 layer ..
.. initialized conv2d_37 layer ..
.. initialized fbn_37 layer ..
.. initialized conv2d_38 layer ..
.. initialized fbn_38 layer ..
.. initialized conv2d_39 layer ..
.. initialized fbn_39 layer ..
.. initialized conv2d_40 layer ..
.. initialized fbn_40 layer ..
.. initialized conv2d_41 layer ..
.. initialized fbn_41 layer ..
.. initialized conv2d_42 layer ..
.. initialized fbn_42 layer ..
.. initialized conv2d_43 layer ..
.. initialized fbn_43 layer ..
.. initialized conv2d_44 layer ..
.. initialized fbn_44 layer ..
.. initialized conv2d_45 layer ..
.. initialized fbn_45 layer ..
.. initialized conv2d_46 layer ..
.. initialized fbn_46 layer ..
.. initialized conv2d_47 layer ..
.. initialized fbn_47 layer ..
.. initialized conv2d_48 layer ..
.. initialized fbn_48 layer ..
.. initialized conv2d_49 layer ..
.. initialized fbn_49 layer ..
.. initialized conv2d_50 layer ..
.. initialized fbn_50 layer ..
.. initialized conv2d_51 layer ..
.. initialized fbn_51 layer ..
.. initialized conv2d_52 layer ..
.. initialized fbn_52 layer ..
.. initialized dense layer ..
step:  100, loss: 0.082, accuracy: 45.188%
step:  200, loss: 0.028, accuracy: 86.094%
step:  300, loss: 0.016, accuracy: 92.250%
step:  400, loss: 0.018, accuracy: 95.094%
step:  500, loss: 0.029, accuracy: 96.844%
[INFERENCE]
loading ./dataset/t10k-images-idx3-ubyte
loaded 10000 items..
loss: 7.108, accuracy: 92.000%
Done.
```

## Features
* Parameter saving and loading
* Network modification
* Learning rate modificiation
* Dataset shuffling
* Debugging
* how ResNet branches forward/backward
* SGD, Adam, RMSprop, Momentum optimizer
* Add more layers(e.g., Dropout, LRN, FusedBatchNormalization, Add, Pad.)

## Reference
- [Professional CUDA C Programming](https://www.wiley.com/en-cn/Professional+CUDA+C+Programming-p-9781118739310)
- [Learn CUDA Programming/Chapter10](https://github.com/PacktPublishing/Learn-CUDA-Programming)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Caffe2](https://github.com/Biu-G/libtorch_source/tree/7ee192cfc2b680ee6e06e5910e90e322e9e7c387/caffe2)
- [NVIDIA CUDNN DOCUMENTATION](https://docs.nvidia.com/deeplearning/cudnn/index.html)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [mnist-cudnn](https://github.com/haanjack/mnist-cudnn)
