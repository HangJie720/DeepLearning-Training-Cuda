#include "layer.h"

#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <cassert>
#include <math.h>
#include <algorithm>

#include <sstream>
#include <fstream>
#include <iostream>

using namespace cudl;

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void PadForward(const int count, const float *in, float *out,
                           const int num, const int channel, const int height_in, const int width_in,
                           const int pad) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        int i = index;  // Preserve the original value
        int height_out = height_in + pad + pad;
        int width_out = width_in + pad + pad;
        int w = i % width_in;
        i /= width_in;
        int h = i % height_in;
        i /= height_in;
        int c = i % channel;
        i /= channel;

        out[((i * channel + c) * height_out + h + pad) * width_out + pad + w] =
                in[index];
        int w1 = index % width_out;
        int h1 = (index / width_out) % height_out;
        if (h1 < pad || h1 > height_out - 1 - pad || w1 < pad || w1 > width_out - 1 - pad) {
            out[index] = 0.f;
        }
    }
}


__global__ void PadBackward(const int count, const float *in, float *out,
                            const int num, const int channel, const int height_in, const int width_in,
                            const int pad) {
    CUDA_1D_KERNEL_LOOP(index, count) {
        int i = index;  // Preserve original value
        int height_out = height_in + pad + pad;
        int width_out = width_in + pad + pad;
        int w = i % width_in;
        i /= width_in;
        int h = i % height_in;
        i /= height_in;
        int c = i % channel;
        i /= channel;
        out[index] = in[((i * channel + c) * height_out + h + pad) *
                        width_out + pad + w];
    }
}

__global__ void adam_update(
        int N,
        int step,
        float *m,
        float *v,
        float *w,
        const float *g,
        const float beta1,
        const float beta2,
        const float eps_hat,
        const float lr) {
    CUDA_1D_KERNEL_LOOP(i, N) {
        // Updating running mean and var.
        m[i] = m[i] * beta1 + g[i] * (1 - beta1);
        v[i] = v[i] * beta2 + g[i] * g[i] * (1 - beta2);
        float mi = m[i] / (1 - std::pow(beta1, step + 1));
        float vi = v[i] / (1 - std::pow(beta2, step + 1));
        // Update parameters.
        w[i] = w[i] - lr * mi / (std::sqrt(vi) + eps_hat);
    }
}

__global__ void momentum_update(
    int N,
    float *m,
    float *w,
    const float *g,
    const float lr,
    const float momentum) {
    CUDA_1D_KERNEL_LOOP(i, N) {
    m[i] = momentum * m[i] + lr * g[i];
        w[i] = w[i] - m[i];
    }
}

__global__ void add(float *array_a, float *array_b, float *array_c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += step) {
        array_c[i] = array_a[i] + array_b[i];
    }
}

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/
Layer::Layer() {
    /* do nothing */
}

Layer::~Layer() {
#if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
    std::cout << "Destroy Layer: " << name_ << std::endl;
#endif

    if (output_ != nullptr) delete output_;
    if (grad_input_ != nullptr) delete grad_input_;

    if (weights_ != nullptr) delete weights_;
    if (biases_ != nullptr) delete biases_;
    if (weights_m_ != nullptr) delete weights_m_;
    if (biases_m_ != nullptr) delete biases_m_;
    if (weights_v_ != nullptr) delete weights_v_;
    if (biases_v_ != nullptr) delete biases_v_;
    if (grad_weights_ != nullptr) delete grad_weights_;
    if (grad_biases_ != nullptr) delete grad_biases_;
}

void Layer::init_weight_bias(unsigned int seed) {
    checkCudaErrors(cudaDeviceSynchronize());

    if (weights_ == nullptr || biases_ == nullptr)
        return;

    // Create random network
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    // He uniform distribution
    float range = sqrt(6.f / input_->size());    // He's initialization
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
        weights_->ptr()[i] = static_cast<float>(dis(gen));
    for (int i = 0; i < biases_->len(); i++)
        biases_->ptr()[i] = 0.f;
    for (int i = 0; i < weights_m_->len(); i++)
        weights_m_->ptr()[i] = 0.f;
    for (int i = 0; i < weights_v_->len(); i++)
        weights_v_->ptr()[i] = 0.f;
    for (int i = 0; i < biases_m_->len(); i++)
        biases_m_->ptr()[i] = 0.f;
    for (int i = 0; i < biases_v_->len(); i++)
        biases_v_->ptr()[i] = 0.f;

    // copy initialized value to the device
    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);
    weights_m_->to(DeviceType::cuda);
    weights_v_->to(DeviceType::cuda);
    biases_m_->to(DeviceType::cuda);
    biases_v_->to(DeviceType::cuda);

    std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}

void Layer::update_weights_biases(float learning_rate) {
    float eps = -1.f * learning_rate;
    if (weights_ != nullptr && grad_weights_ != nullptr) {
#if (DEBUG_UPDATE)
        weights_->print(name_ + "::weights (before update)", true);
        grad_weights_->print(name_ + "::gweights", true);
#endif // DEBUG_UPDATE

        // w = w + eps * dw
        checkCublasErrors(
                cublasSaxpy(cuda_->cublas(),
                            weights_->len(),
                            &eps,
                            grad_weights_->cuda(), 1,
                            weights_->cuda(), 1));

#if (DEBUG_UPDATE)
        weights_->print(name_ + "weights (after update)", true);
        // getchar();
#endif // DEBUG_UPDATE
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
#if (DEBUG_UPDATE)
        biases_->print(name_ + "biases (before update)", true);
        grad_biases_->print(name_ + "gbiases", true);
#endif // DEBUG_UPDATE

        // b = b + eps * db
        checkCublasErrors(
                cublasSaxpy(cuda_->cublas(),
                            biases_->len(),
                            &eps,
                            grad_biases_->cuda(), 1,
                            biases_->cuda(), 1));

#if (DEBUG_UPDATE)
        biases_->print(name_ + "biases (after update)", true);
        // getchar();
#endif // DEBUG_UPDATE
    }
}

void Layer::update_weights_biases_with_adam(float learning_rate, float beta1, float beta2, float eps_hat, int step) {

    if (weights_ != nullptr && grad_weights_ != nullptr) {

#if (DEBUG_UPDATE)
        weights_->print(name_ + "::weights (before update)", true);
        grad_weights_->print(name_ + "::gweights", true);
#endif // DEBUG_UPDATE
        /*
         * mt = b1*mt + (1-b1)*dw
         * vt = b2*vt + (1-b2)*dw*dw
         * mtt = mt/(1-(b1^(i+1)))
         * vtt = vt/(1-(b2^(i+1)))
         * w = w + eps * mtt/(std::sqrt(vtt) + e)
         */
        config = getGpuLaunchConfig(weights_->len(), adam_update, 0, 0);
        adam_update <<< config.block_count, config.thread_per_block >>>
                             (weights_->len(), step, weights_m_->cuda(), weights_v_->cuda(), weights_->cuda(), grad_weights_->cuda(), beta1, beta2, eps_hat, learning_rate);


#if (DEBUG_UPDATE)
        weights_->print(name_ + "weights (after update)", true);
        // getchar();
#endif // DEBUG_UPDATE
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
#if (DEBUG_UPDATE)
        biases_->print(name_ + "biases (before update)", true);
        grad_biases_->print(name_ + "gbiases", true);
#endif // DEBUG_UPDATE
        config = getGpuLaunchConfig(biases_->len(), adam_update, 0, 0);
        adam_update <<<config.block_count,config.thread_per_block>>>
                             (biases_->len(), step, biases_m_->cuda(), biases_v_->cuda(), biases_->cuda(), grad_biases_->cuda(), beta1, beta2, eps_hat, learning_rate);

#if (DEBUG_UPDATE)
        biases_->print(name_ + "biases (after update)", true);
        // getchar();
#endif // DEBUG_UPDATE
    }
}

void Layer::update_weights_biases_with_momentum(float learning_rate, float momentum) {

    if (weights_ != nullptr && grad_weights_ != nullptr) {

#if (DEBUG_UPDATE)
weights_->print(name_ + "::weights (before update)", true);
grad_weights_->print(name_ + "::gweights", true);
#endif // DEBUG_UPDATE
        /*
         * mt = momentum*mt + lr*dw
         * w = w - mt
         */
        config = getGpuLaunchConfig(weights_->len(), momentum_update, 0, 0);
        momentum_update <<< config.block_count, config.thread_per_block >>> (weights_->len(), weights_m_->cuda(), weights_->cuda(), grad_weights_->cuda(), learning_rate, momentum);


#if (DEBUG_UPDATE)
weights_->print(name_ + "weights (after update)", true);
// getchar();
#endif // DEBUG_UPDATE
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
#if (DEBUG_UPDATE)
biases_->print(name_ + "biases (before update)", true);
grad_biases_->print(name_ + "gbiases", true);
#endif // DEBUG_UPDATE
        config = getGpuLaunchConfig(biases_->len(), momentum_update, 0, 0);
        momentum_update <<< config.block_count, config.thread_per_block >>> (biases_->len(), biases_m_->cuda(), biases_->cuda(), grad_biases_->cuda(), learning_rate, momentum);

#if (DEBUG_UPDATE)
biases_->print(name_ + "biases (after update)", true);
// getchar();
#endif // DEBUG_UPDATE
    }
}

float Layer::get_loss(Blob<float> *target) {
    assert("No Loss layer has no loss." && false);
    return EXIT_FAILURE;
}

int Layer::get_accuracy(Blob<float> *target) {
    assert("No Loss layer cannot estimate accuracy." && false);
    return EXIT_FAILURE;
}

int Layer::load_parameter() {
    std::stringstream filename_weights, filename_biases;

    // load weights and biases pretrained parameters
    filename_weights << name_ << ".bin";
    if (weights_->file_read(filename_weights.str()))
        return -1;

    filename_biases << name_ << ".bias.bin";
    if (biases_->file_read(filename_biases.str()))
        return -2;

    std::cout << ".. loaded " << name_ << " pretrain parameter.." << std::endl;

    return 0;
}

int Layer::save_parameter() {
    std::stringstream filename_weights, filename_biases;

    std::cout << ".. saving " << name_ << " parameter ..";

    // Write weights file
    if (weights_) {
        filename_weights << name_ << ".bin";
        if (weights_->file_write(filename_weights.str()))
            return -1;
    }

    // Write bias file
    if (biases_) {
        filename_biases << name_ << ".bias.bin";
        if (biases_->file_write(filename_biases.str()))
            return -2;
    }

    std::cout << " done .." << std::endl;

    return 0;
}

void Layer::set_output_to(Layer *layer) {
    if (output_to_ == nullptr) {
        output_to_ = layer;
    } else {
        copy_output_to_ = layer;
    }
}

void Layer::set_layer_relationship(Layer *input_from, Layer *input2_from) {
    input_from_ = input_from;
    input2_from_ = input2_from;
    if (input_from_ != nullptr) {
        input_from_->set_output_to(this);
    }
    if (input2_from_ != nullptr) {
        input2_from_->set_output_to(this);
    }
}

Blob<float> *Layer::get_input(Blob<float> *input) {
    if (input_from_ != nullptr) {
        input = input_from_->get_output();
    }

    if (input2_from_ != nullptr) {
        input2_ = input2_from_->get_output();
    }
    return input;
}

Blob<float> *Layer::sum_gradients(Blob<float> *grad) {
    Blob<float> *grad2;
    if (output_to_ != nullptr) {
        grad = output_to_->get_grad();
    }

    if (copy_output_to_ != nullptr) {
        grad2 = copy_output_to_->get_grad();
        // grad = grad + grad2
        config = getGpuLaunchConfig(grad2->len(), add, 0, 0);
        add<<<config.block_count,config.thread_per_block>>>(grad->cuda(), grad2->cuda(), grad->cuda(), grad->len());
    }
    return grad;
}

/****************************************************************
 * Dense Layer                                                  *
 ****************************************************************/

Dense::Dense(Layer *input_from, std::string name, int output_size) {
    set_layer_relationship(input_from);

    name_ = name;
    output_size_ = output_size;
}

Dense::~Dense() {
    if (d_one_vec != nullptr)
        cudaFree(d_one_vec);
}

__global__ void init_one_vec(float *d_one_vec, size_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length) return;

    d_one_vec[i] = 1.f;
}

Blob<float> *Dense::forward(Blob<float> *input) {
    input = get_input(input);

    // initialize weights and biases
    if (weights_ == nullptr) {
        // setup parameter size information
        input_size_ = input->c() * input->h() * input->w();

        // initialize weight, bias, and output
        weights_ = new Blob<float>(1, 1, input_size_, output_size_);
        biases_ = new Blob<float>(1, 1, output_size_);
        weights_m_ = new Blob<float>(1, 1, input_size_, output_size_);
        weights_v_ = new Blob<float>(1, 1, input_size_, output_size_);
        biases_m_ = new Blob<float>(1, 1, output_size_);
        biases_v_ = new Blob<float>(1, 1, output_size_);
    }

    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Blob<float>(batch_size_, output_size_);
        else
            output_->reset(batch_size_, output_size_);

        output_->tensor();

        if (d_one_vec != nullptr)
            cudaFree(d_one_vec);
        checkCudaErrors(cudaMalloc((void **) &d_one_vec, sizeof(float) * batch_size_));
        config = getGpuLaunchConfig(batch_size_, init_one_vec, 0, 0);
        init_one_vec <<<config.block_count, config.thread_per_block>>> (d_one_vec, batch_size_);

        // initialize weights and biases
        if (load_pretrain_ && !freeze_) {
            if (load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!freeze_) {
            init_weight_bias();
        } else {
            /* do nothing */
        }
    }

    // output = weights^T * input (without biases)
    checkCublasErrors(cublasSgemm(cuda_->cublas(),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        output_size_, batch_size_, input_size_,
                        &cuda_->one,
                        weights_->cuda(), input_size_,
                        input_->cuda(), input_size_,
                        &cuda_->zero,
                        output_->cuda(), output_size_));

    // output += biases * d_one_vec^T
    checkCublasErrors(cublasSgemm(cuda_->cublas(),
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  output_size_, batch_size_, 1,
                                  &cuda_->one,
                                  biases_->cuda(), output_size_,
                                  d_one_vec, 1,
                                  &cuda_->one,
                                  output_->cuda(), output_size_));


#if (DEBUG_DENSE & 0x01)
    input_->print(  name_ + "::input",  true);
    weights_->print(name_ + "::weight", true);
    biases_->print( name_ + "::bias",   true);
    output_->print( name_ + "::output", true);
#endif // DEBUG_DENSE

    return output_;
}

Blob<float> *Dense::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    if (grad_weights_ == nullptr) {
        grad_weights_ = new Blob<float>(weights_->shape());
        grad_biases_ = new Blob<float>(biases_->shape());
    }

    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }

    // db = (dy) * d_one_vec
    cublasSgemv(cuda_->cublas(),
                CUBLAS_OP_N,
                output_size_, batch_size_,
                &cuda_->one,
                grad_output_->cuda(), output_size_,
                d_one_vec, 1,
                &cuda_->zero,
                grad_biases_->cuda(), 1);

    // dw = x * (dy)^T
    cublasSgemm(cuda_->cublas(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                input_size_, output_size_, batch_size_,
                &cuda_->one,
                input_->cuda(), input_size_,
                grad_output_->cuda(), output_size_,
                &cuda_->zero,
                grad_weights_->cuda(), input_size_);

    // dx = W * dy
    if (!gradient_stop_)
        cublasSgemm(cuda_->cublas(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    input_size_, batch_size_, output_size_,
                    &cuda_->one,
                    weights_->cuda(), input_size_,
                    grad_output_->cuda(), output_size_,
                    &cuda_->zero,
                    grad_input_->cuda(), input_size_);

#if (DEBUG_DENSE & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output->print(  name_ + "::gradients", true, grad_output->n());
    grad_weights_->print(name_ + "::gfilter", true);
    grad_biases_->print( name_ + "::gbias", true);
    if (!gradient_stop_)
        grad_input_->print(  name_ + "::gdata", true);
#endif // DEBUG_DENSE

    return grad_input_;
}

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/

Activation::Activation(Layer *input_from, std::string name, cudnnActivationMode_t mode, float coef) {
    set_layer_relationship(input_from);

    name_ = name;
    mode_ = mode;
    coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, mode, CUDNN_PROPAGATE_NAN, coef);
}

Activation::~Activation() {
    cudnnDestroyActivationDescriptor(act_desc_);
}

Blob<float> *Activation::forward(Blob<float> *input) {
    input = get_input(input);

    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Blob<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();
    }

    cudnnActivationForward(cuda_->cudnn(),
                           act_desc_,
                           &cuda_->one,
                           input_desc_,
                           input->cuda(),
                           &cuda_->zero,
                           output_desc_,
                           output_->cuda());

    return output_;
}

Blob<float> *Activation::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }

    cudnnActivationBackward(cuda_->cudnn(),
                            act_desc_,
                            &cuda_->one,
                            output_desc_, output_->cuda(),
                            output_desc_, grad_output->cuda(),
                            input_desc_, input_->cuda(),
                            &cuda_->zero,
                            input_desc_, grad_input_->cuda());

    return grad_input_;
}

/****************************************************************
 * Softmax definition                                           *
 ****************************************************************/
Softmax::Softmax(Layer *input_from,std::string name) {
    set_layer_relationship(input_from);

    name_ = name;
}

Softmax::~Softmax() {

}

Blob<float> *Softmax::forward(Blob<float> *input) {
    input = get_input(input);

    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Blob<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();
    }

#if (DEBUG_SOFTMAX & 0x01)
        std::cout << name_ << "[FORWARD]" << std::endl;
        input_->print(name_ + "::input", true, input->n());
#endif

    checkCudnnErrors(
            cudnnSoftmaxForward(cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                &cuda_->one, input_desc_, input->cuda(),
                                &cuda_->zero, output_desc_, output_->cuda()));

#if (DEBUG_SOFTMAX & 0x01)
    output_->print(name_ + "::output", true, input->n());
#endif

    return output_;
}

Blob<float> *Softmax::backward(Blob<float> *target) {
    checkCudaErrors(cudaDeviceSynchronize());

    target = sum_gradients(target);

    if (grad_input_ == nullptr || batch_size_ != target->n()) {
        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }

    // set grad_input_ as predict
    checkCudaErrors(cudaMemcpyAsync(grad_input_->cuda(),
                                    output_->cuda(), output_->buf_size(),
                                    cudaMemcpyDeviceToDevice));
    // set grad_input_ = predict - target
    checkCublasErrors(
            cublasSaxpy(cuda_->cublas(), target->len(),
                        &cuda_->minus_one, target->cuda(), 1,
                        grad_input_->cuda(), 1));

    // normalize the grad_output by the batch size
    float scale = 1.f / static_cast<float>(target->n());
    checkCublasErrors(cublasSscal(cuda_->cublas(), target->len(), &scale, grad_input_->cuda(), 1));

#if (DEBUG_SOFTMAX & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    input_->print( name_ + "::input", true);
    output_->print(name_ + "::predict", true);
    target->print( name_ + "::y", true, target->n());
    grad_input_->print(name_ + "::dx", true, target->n());
#endif

    return grad_input_;
}

float Softmax::get_loss(Blob<float> *target) {
    return loss_.loss(output_, target);
}

int Softmax::get_accuracy(Blob<float> *target) {
    int batch_size = output_->n();
    int output_size = output_->size();

    assert(batch_size == target->n());
    assert(output_size == target->size());

    float *h_output, *h_target;
    int idx_output, idx_target;
    int hit_count = 0;

    // get predicts and targets
    h_output = output_->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++) {
        idx_output = 0;
        idx_target = 0;

        for (int i = 1; i < 10; i++) {
            if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
                idx_output = i;
            if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
                idx_target = i;
        }

        if (idx_output == idx_target)
            hit_count++;
    }

    return hit_count;
}

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/

/**
 * Convolutional layer with bias
 */
Conv2D::Conv2D(Layer *input_from,
               std::string name,
               int out_channels,
               int kernel_size,
               int stride,
               int padding,
               int dilation) :
        out_channels_(out_channels),
        kernel_size_(kernel_size),
        stride_(stride),
        padding_(padding),
        dilation_(dilation) {
    set_layer_relationship(input_from);

    name_ = name;

    // create cudnn container handles
    cudnnCreateFilterDescriptor(&filter_desc_);

    cudnnCreateConvolutionDescriptor(&conv_desc_);
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                     padding_, padding_, stride_, stride_, dilation_, dilation_,
                                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
}

Conv2D::~Conv2D() {
    // distroy cudnn container resources
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);

    // terminate internal created blobs
    if (d_workspace != nullptr) cudaFree(d_workspace);
}

void Conv2D::set_workspace() {
    size_t temp_size = 0;

    // forward
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(cuda_->cudnn(),
                                                         input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_fwd_algo_));
    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cuda_->cudnn(),
                                                             input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                             conv_fwd_algo_, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // todo trainable check
    // bwd - filter
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(cuda_->cudnn(),
                                                                input_desc_, output_desc_, conv_desc_, filter_desc_,
                                                                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
                                                                &conv_bwd_filter_algo_));
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda_->cudnn(),
                                                                    input_desc_, output_desc_, conv_desc_, filter_desc_,
                                                                    conv_bwd_filter_algo_, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // bwd - data
    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(cuda_->cudnn(),
                                                              filter_desc_, output_desc_, conv_desc_, input_desc_,
                                                              CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
                                                              &conv_bwd_data_algo_));
    checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->cudnn(),
                                                                  filter_desc_, output_desc_, conv_desc_, input_desc_,
                                                                  conv_bwd_data_algo_, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    if (workspace_size > 0) {
        if (d_workspace != nullptr) checkCudaErrors(cudaFree(d_workspace));
        checkCudaErrors(cudaMalloc((void **) &d_workspace, workspace_size));
    }
}

Blob<float> *Conv2D::forward(Blob<float> *input) {
    input = get_input(input);

    // initialize weights and bias
    if (weights_ == nullptr) {
        // initialize containers handles
        checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_desc_,
                                                    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                    out_channels_, input->c(), kernel_size_, kernel_size_));

        weights_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        biases_ = new Blob<float>(1, out_channels_);    // bias size
        bias_desc_ = biases_->tensor();
        weights_m_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        weights_v_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        biases_m_ = new Blob<float>(1, out_channels_);
        biases_v_ = new Blob<float>(1, out_channels_);
    }

    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->n()) {
        // initialize input
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        // initilaize output
        checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(
                conv_desc_, input_desc_, filter_desc_,
                &output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]));
        if (output_ == nullptr)
            output_ = new Blob<float>(output_size_);
        else
            output_->reset(output_size_);

        output_desc_ = output_->tensor();

        // initialize workspace for cudnn
        set_workspace();

        // initialize weights
        if (load_pretrain_ && !freeze_) {
            if (load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!freeze_) {
            init_weight_bias();
        } else {
            /* do nothing */
        }
    }

    checkCudnnErrors(cudnnConvolutionForward(cuda_->cudnn(),
                                             &cuda_->one, input_desc_, input_->cuda(),
                                             filter_desc_, weights_->cuda(), conv_desc_, conv_fwd_algo_, d_workspace,
                                             workspace_size,
                                             &cuda_->zero, output_desc_, output_->cuda()));

    checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(),
                                    &cuda_->one, bias_desc_, biases_->cuda(),
                                    &cuda_->one, output_desc_, output_->cuda()));

#if (DEBUG_CONV & 0x01)
    input_->print(  name_ + "::input", true, input_->n(), 28);
    weights_->print(name_ + "::weight", true);
    biases_->print( name_ + "::bias", true);
    output_->print( name_ + "::output", true);
#endif

    return output_;
}

Blob<float> *Conv2D::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    // initialize grad_output back-propagation space
    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;
        grad_weights_ = new Blob<float>(weights_->shape());
        grad_biases_ = new Blob<float>(1, biases_->c());

        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }

    // gradients of biases
    checkCudnnErrors(
            cudnnConvolutionBackwardBias(cuda_->cudnn(),
                                         &cuda_->one,
                                         output_desc_, grad_output->cuda(),
                                         &cuda_->zero,
                                         bias_desc_, grad_biases_->cuda()));

    // gradients of weights
    checkCudnnErrors(
            cudnnConvolutionBackwardFilter(cuda_->cudnn(),
                                           &cuda_->one,
                                           input_desc_, input_->cuda(),
                                           output_desc_, grad_output_->cuda(),
                                           conv_desc_, conv_bwd_filter_algo_, d_workspace, workspace_size,
                                           &cuda_->zero,
                                           filter_desc_, grad_weights_->cuda()));

    // gradients of input data
    if (!gradient_stop_) checkCudnnErrors(
            cudnnConvolutionBackwardData(cuda_->cudnn(),
                                         &cuda_->one,
                                         filter_desc_, weights_->cuda(),
                                         output_desc_, grad_output->cuda(),
                                         conv_desc_, conv_bwd_data_algo_, d_workspace, workspace_size,
                                         &cuda_->zero,
                                         input_desc_, grad_input_->cuda()));

#if (DEBUG_CONV & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output->print( name_ + "::gradients", true);
    grad_biases_->print(name_ + "gbias", true);
    grad_weights_->print(name_+ "gfilter", true);
    if (!gradient_stop_)
        grad_input_->print(name_+"gdata", true);
#endif

#if (DEBUG_CONV & 0x04)
    grad_output->print( name_ + "::gradients", true);
    grad_biases_->print( name_ + "::gbias", true);
#endif

    return grad_input_;
}

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/

Pooling::Pooling(Layer *input_from,
                 std::string name,
                 int kernel_size,
                 int padding,
                 int stride,
                 cudnnPoolingMode_t mode) :
        kernel_size_(kernel_size),
        padding_(padding),
        stride_(stride),
        mode_(mode) {
    set_layer_relationship(input_from);

    name_ = name;

    cudnnCreatePoolingDescriptor(&pool_desc_);
    cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
                                kernel_size_, kernel_size_, padding_, padding_, stride_, stride_);
}

Pooling::~Pooling() {
    cudnnDestroyPoolingDescriptor(pool_desc_);
}

Blob<float> *Pooling::forward(Blob<float> *input) {
    input = get_input(input);

    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;

        // resource initialize
        input_desc_ = input_->tensor();
        batch_size_ = input->n();

        // setting output
        cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_,
                                          &output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]);
        if (output_ == nullptr)
            output_ = new Blob<float>(output_size_);
        else
            output_->reset(output_size_);

        output_desc_ = output_->tensor();
    }

    cudnnPoolingForward(cuda_->cudnn(), pool_desc_,
                        &cuda_->one, input_desc_, input_->cuda(),
                        &cuda_->zero, output_desc_, output_->cuda());

    return output_;
}

Blob<float> *Pooling::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }

    checkCudnnErrors(
            cudnnPoolingBackward(cuda_->cudnn(), pool_desc_,
                                 &cuda_->one,
                                 output_desc_, output_->cuda(),
                                 output_desc_, grad_output->cuda(),
                                 input_desc_, input_->cuda(),
                                 &cuda_->zero,
                                 input_desc_, grad_input_->cuda()));

    return grad_input_;
}


/****************************************************************
 * Add definition                                               *
 ****************************************************************/
Add::Add(Layer *input_from, Layer *input2_from, std::string name) {
    set_layer_relationship(input_from, input2_from);

    name_ = name;
}

Add::~Add() {}

Blob<float> *Add::forward(Blob<float> *input) {
    input = get_input(input);

    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Blob<float>(input->shape());
        else
            output_->reset(input->shape());
        config = getGpuLaunchConfig(input_->len(), add, 0, 0);
    }

    // y = x + x2
    add<<<config.block_count, config.thread_per_block>>>(input_->cuda(), input2_->cuda(), output_->cuda(), input_->len());

#if (DEBUG_ADD & 0x01)
    std::cout << name_ << "[FORWARD]" << std::endl;
    input_->print(  name_ + "::input", true, input_->n());
    input2_->print(  name_ + "::input", true, input2_->n());
    output_->print( name_ + "::output", true);
#endif

    return output_;
}

Blob<float> *Add::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    // initialize grad_output back-propagation space
    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr) {
            grad_input_ = new Blob<float>(input_->shape());
        } else {
            grad_input_->reset(input_->shape());
        }
    }
    // y = x + x2
    // dy/dx = dy/dx2 = 1
    checkCudaErrors(cudaMemcpy(grad_input_->cuda(),
                               grad_output_->cuda(), grad_output_->buf_size(),
                               cudaMemcpyDeviceToDevice));

#if (DEBUG_ADD & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output_->print(  name_ + "::gradients", true, grad_output_->n());
    if (!gradient_stop_)
        grad_input_->print(  name_ + "::gdata", true);
#endif // DEBUG_PADDING

    return grad_input_;
}

/****************************************************************
 * Pad definition                                               *
 ****************************************************************/
Pad::Pad(Layer *input_from, std::string name, std::array<int, 8> paddings, int pad_value) {
    set_layer_relationship(input_from);

    name_ = name;
    paddings_ = paddings;
    pad_value_ = pad_value;
}

Pad::~Pad() {}

Blob<float> *Pad::forward(Blob<float> *input) {
    input = get_input(input);

    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        batch_size_ = input_->n();

        int n = input_->n() + paddings_.at(0) + paddings_.at(1);
        int c = input_->c() + paddings_.at(2) + paddings_.at(3);
        int h = input_->h() + paddings_.at(4) + paddings_.at(5);
        int w = input_->w() + paddings_.at(6) + paddings_.at(7);
        if (output_ == nullptr)
            output_ = new Blob<float>(n, c, h, w);
        else
            output_->reset(n, c, h, w);
    }
    // eigen implemented.
     Eigen::GpuStreamDevice stream;
     Eigen::GpuDevice gpu_device(&stream);
     Eigen::TensorMap <Eigen::Tensor<float, 4, Eigen::RowMajor>> gpu_in(input_->cuda(), input_->n(), input_->c(),
	 	                                      input_->h(), input_->w());
     Eigen::TensorMap <Eigen::Tensor<float, 4, Eigen::RowMajor>> gpu_out(output_->cuda(), output_->n(), output_->c(),
	 	                                      output_->h(), output_->w());
     Eigen::array<std::pair<int, int>, 4> pads;
     pads[0] = std::make_pair(paddings_.at(0), paddings_.at(1));
     pads[1] = std::make_pair(paddings_.at(2), paddings_.at(3));
     pads[2] = std::make_pair(paddings_.at(4), paddings_.at(5));
     pads[3] = std::make_pair(paddings_.at(6), paddings_.at(7));
     gpu_out.device(gpu_device) = gpu_in.pad(pads, pad_value_);

    // cuda implemented.
    //config = getGpuLaunchConfig(input_->len(), PadForward, 0, 0);
    //PadForward <<<config.block_count, config.thread_per_block >>>(input_->len(), input_->cuda(), output_->cuda(), input_->n(), input_->c(), input_->h(), input_->w(), paddings_.at(4));

    //cudaDeviceSynchronize();

#if (DEBUG_PADDING & 0x01)
    std::cout << name_ << "[FORWARD]" << std::endl;
    input_->print(  name_ + "::input", true, input_->n(), output_->h());
    output_->print( name_ + "::output", true, output_->n(), output_->h());
#endif

    return output_;
}

Blob<float> *Pad::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    // initialize grad_output back-propagation space
    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;
        batch_size_ = grad_output_->n();

	int n = grad_output_->n() - paddings_.at(0) - paddings_.at(1);
	int c = grad_output_->c() - paddings_.at(2) - paddings_.at(3);
	int h = grad_output_->h() - paddings_.at(4) - paddings_.at(5);
	int w = grad_output_->w() - paddings_.at(6) - paddings_.at(7);
        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(n, c, h, w);
        else
            grad_input_->reset(n, c, h, w);
        config = getGpuLaunchConfig(grad_input_->len(), PadBackward, 0, 0);
    }

    if (!gradient_stop_) {
        // eigen implemented.
         Eigen::GpuStreamDevice stream;
         Eigen::GpuDevice gpu_device(&stream);
         Eigen::TensorMap <Eigen::Tensor<float, 4, Eigen::RowMajor>> gpu_in(grad_output_->cuda(),
	                                                   grad_output_->n(), grad_output_->c(),
                                                           grad_output_->h(), grad_output_->w());
         Eigen::TensorMap <Eigen::Tensor<float, 4, Eigen::RowMajor>> gpu_out(grad_input_->cuda(),
		 	                                   grad_input_->n(), grad_input_->c(),
                                                            grad_input_->h(), grad_input_->w());
         Eigen::array<int, 4> offsets = {paddings_.at(0), paddings_.at(2), paddings_.at(4), paddings_.at(6)};
         Eigen::array<int, 4> extents = {grad_input_->n(), grad_input_->c(), grad_input_->h(), grad_input_->w()};
         gpu_out.device(gpu_device) = gpu_in.slice(offsets, extents);

        // cuda implemented.
        //PadBackward <<<config.block_count, config.thread_per_block>>>(grad_input_->len(), grad_output_->cuda(), grad_input_->cuda(), grad_output_->n(), grad_output_->c(), grad_output_->h(), grad_output_->w(), paddings_.at(4));
        //cudaDeviceSynchronize();
    }

#if (DEBUG_PADDING & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output_->print(  name_ + "::gradients", true, grad_output_->n());
    if (!gradient_stop_)
        grad_input_->print(  name_ + "::gdata", true);
#endif // DEBUG_PADDING

    return grad_input_;
}

/****************************************************************
 * FusedBatchNormalization definition                           *
 ****************************************************************/
FusedBatchNormalization::FusedBatchNormalization(Layer *input_from, std::string name, cudnnBatchNormMode_t mode) {
    set_layer_relationship(input_from);

    name_ = name;
    mode_ = mode;

    checkCudnnErrors(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc_));
}

FusedBatchNormalization::~FusedBatchNormalization() {
    cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc_);
}

Blob<float> *FusedBatchNormalization::forward(Blob<float> *input) {
    input = get_input(input);

    // initialize weights and biases
    if (weights_ == nullptr) {
        // initialize weight, bias
        size_ = input->c();
        weights_ = new Blob<float>(1, size_, 1, 1);
        biases_ = new Blob<float>(1, size_, 1, 1);
        weights_m_ = new Blob<float>(1, size_, 1, 1);
        weights_v_ = new Blob<float>(1, size_, 1, 1);
        biases_m_ = new Blob<float>(1, size_, 1, 1);
        biases_v_ = new Blob<float>(1, size_, 1, 1);
    }
    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        batch_size_ = input->n();
        input_desc_ = input_->tensor();

        if (output_ == nullptr)
            output_ = new Blob<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();

        cudaMalloc(&resultRunningMean_, sizeof(float) * size_);
        cudaMalloc(&resultRunningVariance_, sizeof(float) * size_);

        cudaMalloc(&resultSaveMean_, sizeof(float) * size_);
        cudaMalloc(&resultSaveInvVariance_, sizeof(float) * size_);

        checkCudnnErrors(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc_, input_desc_, mode_));

        // initialize weights and biases
        if (load_pretrain_ && !freeze_) {
            if (load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!freeze_) {
            init_weight_bias();
        } else {
            /* do nothing */
        }
    }

    //y = beta*y + alpha *[bnBias + (bnScale * (x-estimatedMean)/sqrt(epsilon + estimatedVariance)]
    checkCudnnErrors(
            cudnnBatchNormalizationForwardTraining(cuda_->cudnn(), mode_, &cuda_->one, &cuda_->zero, input_desc_,
                                                   input_->cuda(), output_desc_, output_->cuda(),
                                                   bnScaleBiasMeanVarDesc_, weights_->cuda(), biases_->cuda(),
                                                   cuda_->one, resultRunningMean_, resultRunningVariance_,
                                                   CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveInvVariance_));


#if (DEBUG_FBN & 0x01)
    std::cout << name_ << "[FORWARD]" << std::endl;
    input_->print(  name_ + "::input", true, input_->n(), input_->h());
    weights_->print(name_ + "::weight", true, weights_->n(), weights_->c());
    biases_->print( name_ + "::bias", true, biases_->n(), biases_->c());
    output_->print( name_ + "::output", true, output_->n(), output_->h());
#endif

    return output_;
}

Blob<float> *FusedBatchNormalization::backward(Blob<float> *grad_output) {
    grad_output = sum_gradients(grad_output);

    // initialize grad_output back-propagation space
    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;
        grad_weights_ = new Blob<float>(weights_->shape());
        grad_biases_ = new Blob<float>(biases_->shape());

        if (grad_input_ == nullptr)
            grad_input_ = new Blob<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }

    checkCudnnErrors(
            cudnnBatchNormalizationBackward(cuda_->cudnn(), mode_, &cuda_->one, &cuda_->zero, &cuda_->one, &cuda_->zero,
                                            input_desc_, input_->cuda(), output_desc_, grad_output_->cuda(), input_desc_,
                                            grad_input_->cuda(), bnScaleBiasMeanVarDesc_, weights_->cuda(), grad_weights_->cuda(),
                                            grad_biases_->cuda(), CUDNN_BN_MIN_EPSILON, resultSaveMean_, resultSaveInvVariance_));

#if (DEBUG_FBN & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output_->print(  name_ + "::gradients", true, grad_output_->n());
    grad_weights_->print(name_ + "::gfilter", true);
    grad_biases_->print( name_ + "::gbias", true);
    if (!gradient_stop_)
        grad_input_->print(  name_ + "::gdata", true);
#endif // DEBUG_FBN

    return grad_input_;
}
