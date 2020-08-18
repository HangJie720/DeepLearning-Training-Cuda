// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_USE_GPU

//#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

#define EIGEN_GPU_TEST_C99_MATH  EIGEN_HAS_CXX11

#include <iostream>
using Eigen::Tensor;

void test_gpu_nullary() {
  Tensor<float, 1, 0, int> in1(2);
  Tensor<float, 1, 0, int> in2(2);
  in1.setRandom();
  in2.setRandom();

  std::size_t tensor_bytes = in1.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  gpuMalloc((void**)(&d_in1), tensor_bytes);
  gpuMalloc((void**)(&d_in2), tensor_bytes);
  gpuMemcpy(d_in1, in1.data(), tensor_bytes, gpuMemcpyHostToDevice);
  gpuMemcpy(d_in2, in2.data(), tensor_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in1(
      d_in1, 2);
  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in2(
      d_in2, 2);

  gpu_in1.device(gpu_device) = gpu_in1.constant(3.14f);
  gpu_in2.device(gpu_device) = gpu_in2.random();

  Tensor<float, 1, 0, int> new1(2);
  Tensor<float, 1, 0, int> new2(2);

  assert(gpuMemcpyAsync(new1.data(), d_in1, tensor_bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);
  assert(gpuMemcpyAsync(new2.data(), d_in2, tensor_bytes, gpuMemcpyDeviceToHost,
                         gpu_device.stream()) == gpuSuccess);

  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  for (int i = 0; i < 2; ++i) {
//    VERIFY_IS_APPROX(new1(i), 3.14f);
//    VERIFY_IS_NOT_EQUAL(new2(i), in2(i));
      std::cout << new1(i) << std::endl;
      std::cout << "3.14f" << std::endl;
      std::cout << new2(i) << std::endl;
      std::cout << in2(i) << std::endl;
      std::cout << "-----------------" << std::endl;
  }

  gpuFree(d_in1);
  gpuFree(d_in2);
}


int main(int argc, char* argv[]){
    test_gpu_nullary();
    return 0;
}
