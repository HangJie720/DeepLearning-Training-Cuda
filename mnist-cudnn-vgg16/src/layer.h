#ifndef _LAYER_H_
#define _LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "loss.h"
#include "helper.h"

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

        // output memory
        Blob<float> *input_ = nullptr;          /* x   */
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

        void update_weights_biases_with_rmsprop(float learning_rate, float decay, float eps_hat);

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
        Dense(std::string name, int out_size);

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
        Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);

        ~Activation();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

        Blob<float> *backward_relu(Blob<float> *grad_input, Blob<float> *grad_input2);

    private:
        cudnnActivationDescriptor_t act_desc_;
        cudnnActivationMode_t mode_;
        float coef_;
    };

    class Softmax : public Layer {
    public:
        Softmax(std::string name);

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
        Conv2D(std::string name,
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
        Pooling(std::string name,
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


    class LRN : public Layer {
    public:
        LRN(std::string name, unsigned lrnN = 5, double lrnAlpha = 0.0001, double lrnBeta = 0.75, double lrnK = 1.0);

        ~LRN();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        unsigned lrnN_;
        double lrnAlpha_;
        double lrnBeta_;
        double lrnK_;
        cudnnLRNDescriptor_t normDesc_;
    };

    class Dropout : public Layer {
    public:
        Dropout(std::string name, float dropout = 0.5);

        ~Dropout();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        float dropout_;
        size_t stateSize_ = 0;
        void *states_ = nullptr;
        unsigned long long seed_ = 1337ull; // Pick a seed.
        size_t reserveSize_ = 0;
        void *m_pReserve_ = nullptr;
        cudnnDropoutDescriptor_t dropoutDesc_;
    };

    class FusedBatchNormalization : public Layer {
    public:
        FusedBatchNormalization(std::string name, cudnnBatchNormMode_t mode);

        ~FusedBatchNormalization();

        Blob<float> *forward(Blob<float> *input);

        Blob<float> *backward(Blob<float> *grad_input);

    private:
        int size_;
        int batch_count_ = 0;
        double epison = 0.001;
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