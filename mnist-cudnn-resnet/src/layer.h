#ifndef _LAYER_H_
#define _LAYER_H_
#define EIGEN_USE_GPU

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "loss.h"
#include "helper.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

using Eigen::Tensor;

namespace cudl {

    class Layer {
    public:
        Layer();

        ~Layer();

        virtual Blob<float> *forward(Blob<float> *input) = 0;

        virtual Blob<float> *backward(Blob<float> *grad_input) = 0;

        std::string get_name() { return name_; }

        virtual float get_loss(Blob<float> *target);

        virtual int get_accuracy(Blob<float> *target);

        void set_cuda_context(CudaContext *context) { cuda_ = context; }

        void set_load_pretrain() { load_pretrain_ = true; };

        void set_gradient_stop() { gradient_stop_ = true; }

        void set_output_to(Layer *layer);
        void set_layer_relationship(Layer *input_from, Layer *input2_from = nullptr);
        Blob<float> *get_output() { return output_; }
        Blob<float> *get_grad() { return grad_input_; }
        Blob<float> *get_input(Blob<float> *input);
	    Blob<float> *sum_gradients(Blob<float> *grad);

        /* Weight Freeze or Unfreeze */
        void freeze() { freeze_ = true; }

        void unfreeze() { freeze_ = false; }

    protected:
        // name of layer
        std::string name_;

        // tensor descriptor for the input/output tensors
        cudnnTensorDescriptor_t input_desc_;
        cudnnTensorDescriptor_t output_desc_;
        // weight/bias descriptor
        cudnnFilterDescriptor_t filter_desc_;
        cudnnTensorDescriptor_t bias_desc_;

        // relationship with other layer
        Layer *input_from_ = nullptr;
        Layer *input2_from_ = nullptr;
        Layer *output_to_ = nullptr;
        Layer *copy_output_to_ = nullptr;

        // input/output memory
        Blob<float> *input_ = nullptr;          /* x   */
        Blob<float> *input2_ = nullptr;         /* x2  */
        Blob<float> *output_ = nullptr;         /* y   */
        Blob<float> *grad_input_ = nullptr;     /* dx  */
        Blob<float> *grad_output_ = nullptr;    /* dy  */

        // master weights & bias
        bool freeze_ = false;     /* control parameter updates */
        Blob<float> *weights_ = nullptr;        /* w */
        Blob<float> *weights_m_ = nullptr;      /* wm */
        Blob<float> *weights_v_ = nullptr;      /* wv */
        Blob<float> *biases_ = nullptr;         /* b */
        Blob<float> *biases_m_ = nullptr;       /* bm */
        Blob<float> *biases_v_ = nullptr;       /* bv */
        Blob<float> *grad_weights_ = nullptr;   /* dw */
        Blob<float> *grad_biases_ = nullptr;    /* db */

        int batch_size_ = 0;  // mini-batch size

        // initialize weights along with the input size
        void init_weight_bias(unsigned int seed = 0);

        void update_weights_biases(float learning_rate);

        void update_weights_biases_with_adam(float learning_rate, float beta1, float beta2, float eps_hat, int step);

        void update_weights_biases_with_momentum(float learning_rate, float momentum);

        // cuda handle container
        CudaContext *cuda_ = nullptr;

        // pretrain parameters
        bool load_pretrain_ = false;

        int load_parameter();

        int save_parameter();

        // gradient stop tagging
        bool gradient_stop_ = false;

        friend class Network;
    };

    class Dense : public Layer {
    public:
        Dense(Layer *input_from, std::string name, int out_size);

        ~Dense();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        int input_size_ = 0;
        int output_size_ = 0;

        float *d_one_vec = nullptr;
    };

    class Activation : public Layer {
    public:
        Activation(Layer *input_from, std::string name, cudnnActivationMode_t mode, float coef = 0.f);

        ~Activation();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        cudnnActivationDescriptor_t act_desc_;
        cudnnActivationMode_t mode_;
        float coef_;
    };

    class Softmax : public Layer {
    public:
        Softmax(Layer *input_from, std::string name);

        ~Softmax();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

        float get_loss(Blob<float> *target);

        int get_accuracy(Blob<float> *target);

    private:
        CrossEntropyLoss loss_;
    };

    class Conv2D : public Layer {
    public:
        Conv2D(Layer *input_from,
               std::string name,
               int out_channels,
               int kernel_size,
               int stride = 1,
               int padding = 0,
               int dilation = 1);

        ~Conv2D();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_output);

    private:
        int out_channels_;
        int kernel_size_;
        int stride_;
        int padding_;
        int dilation_;

        std::array<int, 4> output_size_;

        // convolution
        cudnnConvolutionDescriptor_t conv_desc_;

        cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
        cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;
        cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;

        size_t workspace_size = 0;
        void **d_workspace = nullptr;

        void set_workspace();
    };

    class Pooling : public Layer {
    public:
        Pooling(Layer *input_from,
                std::string name,
                int kernel_size,
                int padding,
                int stride,
                cudnnPoolingMode_t mode);

        ~Pooling();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_output);

    private:
        int kernel_size_;
        int padding_;
        int stride_;
        cudnnPoolingMode_t mode_;

        std::array<int, 4> output_size_;
        cudnnPoolingDescriptor_t pool_desc_;
    };

    class Add : public Layer {
    public:
        Add(Layer *input_from, Layer *input2_from, std::string name);

        ~Add();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    };

    class Pad : public Layer {
    public:
        Pad(Layer *input_from, std::string name, std::array<int, 8> paddings, int pad_value);

        ~Pad();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        std::array<int, 8> paddings_;
        int pad_value_ = 0;
    };

    class FusedBatchNormalization : public Layer {
    public:
        FusedBatchNormalization(Layer *input_from, std::string name, cudnnBatchNormMode_t mode);

        ~FusedBatchNormalization();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        int size_;
        int batch_count_ = 0;
        double epison = 0.00001;
        double exponentialAverageFactor_ = 0;

        float *resultRunningMean_ = nullptr;
        float *resultRunningVariance_ = nullptr;
        float *resultSaveMean_ = nullptr;
        float *resultSaveInvVariance_ = nullptr;

        cudnnBatchNormMode_t mode_;
        cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_;
    };

} // namespace cudl

#endif // _LAYER_H_
