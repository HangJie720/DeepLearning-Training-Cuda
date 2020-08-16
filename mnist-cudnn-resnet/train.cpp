#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvToolsExt.h>

#define SWITCH_CHAR             '-'
using namespace cudl;

int main(int argc, char *argv[]) {
    /* configure the network */
    int batch_size_train = 32;
    int num_steps_train = 2000;
    int monitoring_step = 100;

    float lr_decay = 0.00005f;
    float learning_rate = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps_hat = 0.00000001f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 1000;
    int num_steps_test = 10;

    int error = 0;
    argc -= 1;
    argv++;
    while (argc) {
        if (*argv[0] == SWITCH_CHAR) {
            switch (*(argv[0] + 1)) {
                case 'b':
                    batch_size_train = atol(argv[0] + 2);
                    break;
                case 'n':
                    num_steps_train = atol(argv[0] + 2);
                    break;
                case 'm':
                    monitoring_step = atol(argv[0] + 2);
                    break;
                case 'l':
                    load_pretrain = true;
                    break;
                case 's':
                    file_save = true;
                    break;
                case 'x':
                    batch_size_test = atol(argv[0] + 2);
                    break;
                case 'y':
                    num_steps_test = atol(argv[0] + 2);
                    break;
                default:
                    error++;
                    break;
            }
            if (error) {
                fprintf(stderr, "Unknown switch '%c%s'\n\n", SWITCH_CHAR, argv[0] + 1);
                return error;
            }
        } else {
            fprintf(stderr, "Invalid separator '%c' for option '%s'\n\n", *argv[0], argv[0]);
            return 1;
        }
        argc -= 1;
        argv++;
    }

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, true);

    // step 2. model initialization
    Network model;
    Layer *mainline = nullptr, *shortcut = nullptr;

    //mainline = model.add_layer(new Pad(mainline, "resnet_model_Pad", {0, 0, 0, 0, 3, 3, 3, 3}, 0)); //[1,1,28,28] -> [1,1,34,34]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_Conv2D", 64, 7, 2, 3)); //[1,1,34,34] -> [1,64,14,14]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu", CUDNN_ACTIVATION_RELU)); //[1,64,14,14] -> [1,64,14,14]
    mainline = model.add_layer(new Pooling(mainline, "resnet_model_max_pooling2d_MaxPool", 3, 1, 2, CUDNN_POOLING_MAX)); //[1,64,14,14] -> [1,64,7,7]

    // block1 start
    shortcut = mainline;
    shortcut = model.add_layer(new Conv2D(shortcut, "resnet_model_conv2d_1_Conv2D", 256, 1, 1, 0)); //[1,64,7,7] -> [1,256,7,7]
    shortcut = model.add_layer(new FusedBatchNormalization(shortcut, "resnet_model_batch_normalization_1_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_2_Conv2D", 64, 1, 1, 0)); //[1,64,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_2_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_1", CUDNN_ACTIVATION_RELU)); //[1,64,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_3_Conv2D", 64, 3, 1, 1)); //[1,64,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_3_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_2", CUDNN_ACTIVATION_RELU)); //[1,64,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_4_Conv2D", 256, 1, 1, 0)); //[1,64,7,7] -> [1,256,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_4_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add")); //[1,256,7,7] -> [1,256,7,7]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_3", CUDNN_ACTIVATION_RELU));//[1,256,7,7] -> [1,256,7,7]

    shortcut = mainline;
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_5_Conv2D", 64, 1, 1, 0));//[1,256,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_5_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_4", CUDNN_ACTIVATION_RELU));//[1,64,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_6_Conv2D", 64, 3, 1, 1));//[1,256,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_6_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_5", CUDNN_ACTIVATION_RELU));//[1,64,7,7] -> [1,64,7,7]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_7_Conv2D", 256, 1, 1, 0));//[1,64,7,7] -> [1,256,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_7_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_1")); //[1,256,7,7] -> [1,256,7,7]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_6", CUDNN_ACTIVATION_RELU));//[1,256,7,7] -> [1,256,7,7]
    // block1 end

    // block2 start
    shortcut = mainline;
    //shortcut = model.add_layer(new Pad(shortcut, "resnet_model_Pad_1", {0, 0, 0, 0, 0, 0, 0, 0}, 0));//[1,256,7,7] -> [1,256,7,7]
    shortcut = model.add_layer(new Conv2D(shortcut, "resnet_model_conv2d_8_Conv2D", 512, 1, 2));//[1,256,7,7] -> [1,512,4,4]
    shortcut = model.add_layer(new FusedBatchNormalization(shortcut, "resnet_model_batch_normalization_8_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_9_Conv2D", 128, 1, 1, 0));//[1,256,7,7] -> [1,128,7,7]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_9_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_7", CUDNN_ACTIVATION_RELU));//[1,128,7,7] -> [1,128,7,7]
    //mainline = model.add_layer(new Pad(mainline, "resnet_model_Pad_2", {0, 0, 0, 0, 1, 1, 1, 1}, 0));//[1,128,7,7] -> [1,128,9,9]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_10_Conv2D", 128, 3, 2, 1));//[1,128,9,9] -> [1,128,4,4]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_10_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_8", CUDNN_ACTIVATION_RELU));//[1,128,4,4] -> [1,128,4,4]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_11_Conv2D", 512, 1, 1, 0));//[1,128,4,4] -> [1,512,4,4]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_11_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_2")); //[1,512,4,4] -> [1,512,4,4]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_9", CUDNN_ACTIVATION_RELU));//[1,512,4,4] -> [1,512,4,4]

    shortcut = mainline;
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_12_Conv2D", 128, 1, 1, 0));//[1,512,4,4] -> [1,128,4,4]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_12_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_10", CUDNN_ACTIVATION_RELU));//[1,128,4,4] -> [1,128,4,4]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_13_Conv2D", 128, 3, 1, 1));//[1,128,4,4] -> [1,128,4,4]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_13_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_11", CUDNN_ACTIVATION_RELU));//[1,128,4,4] -> [1,128,4,4]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_14_Conv2D", 512, 1, 1, 0));//[1,128,4,4] -> [1,512,4,4]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_14_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_3")); //[1,512,4,4] -> [1,512,4,4]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_12", CUDNN_ACTIVATION_RELU));//[1,512,4,4] -> [1,512,4,4]
    // block2 end

    // block3 start
    shortcut = mainline;
    //shortcut = model.add_layer(new Pad(shortcut, "resnet_model_Pad_3", {0, 0, 0, 0, 0, 0, 0, 0}, 0));//[1,512,4,4] -> [1,512,4,4]
    shortcut = model.add_layer(new Conv2D(shortcut, "resnet_model_conv2d_15_Conv2D", 1024, 1, 2));//[1,512,4,4] -> [1,1024,2,2]
    shortcut = model.add_layer(new FusedBatchNormalization(shortcut, "resnet_model_batch_normalization_15_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_16_Conv2D", 256, 1, 1, 0));//[1,512,4,4] -> [1,256,4,4]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_16_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_13", CUDNN_ACTIVATION_RELU));//[1,256,4,4] -> [1,256,4,4]
    //mainline = model.add_layer(new Pad(mainline, "resnet_model_Pad_4", {0, 0, 0, 0, 1, 1, 1, 1}, 0));//[1,256,4,4] -> [1,256,6,6]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_17_Conv2D", 256, 3, 2, 1));//[1,256,6,6] -> [1,256,2,2]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_17_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_14", CUDNN_ACTIVATION_RELU));//[1,256,2,2] -> [1,256,2,2]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_18_Conv2D", 1024, 1, 1, 0));//[1,256,2,2] -> [1,1024,2,2]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_18_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_4")); //[1,1024,2,2] -> [1,1024,2,2]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_15", CUDNN_ACTIVATION_RELU));//[1,1024,2,2] -> [1,1024,2,2]

    shortcut = mainline;
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_19_Conv2D", 256, 1, 1, 0));//[1,1024,2,2] -> [1,256,2,2]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_19_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_16", CUDNN_ACTIVATION_RELU));//[1,256,2,2] -> [1,256,2,2]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_20_Conv2D", 256, 3, 1, 1));//[1,256,2,2] -> [1,256,2,2]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_20_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_17", CUDNN_ACTIVATION_RELU));//[1,256,2,2] -> [1,256,2,2]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_21_Conv2D", 1024, 1, 1, 0));//[1,256,2,2] -> [1,1024,2,2]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_21_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_5")); //[1,1024,2,2] -> [1,1024,2,2]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_18", CUDNN_ACTIVATION_RELU));//[1,1024,2,2] -> [1,1024,2,2]
    // block3 end

    // block4 start
    shortcut = mainline;
    //shortcut = model.add_layer(new Pad(shortcut, "resnet_model_Pad_5", {0, 0, 0, 0, 0, 0, 0, 0}, 0));//[1,1024,2,2] -> [1,1024,2,2]
    shortcut = model.add_layer(new Conv2D(shortcut, "resnet_model_conv2d_22_Conv2D", 2048, 1, 2));//[1,1024,2,2] -> [1,2048,1,1]
    shortcut = model.add_layer(new FusedBatchNormalization(shortcut, "resnet_model_batch_normalization_22_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_23_Conv2D", 512, 1, 1, 0));//[1,1024,2,2] -> [1,512,2,2]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_23_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_19", CUDNN_ACTIVATION_RELU));//[1,512,2,2] -> [1,512,2,2]
    //mainline = model.add_layer(new Pad(mainline, "resnet_model_Pad_6", {0, 0, 0, 0, 1, 1, 1, 1}, 0));//[1,512,2,2] -> [1,512,4,4]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_24_Conv2D", 512, 3, 2, 1));//[1,512,4,4] -> [1,512,1,1]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_24_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_20", CUDNN_ACTIVATION_RELU));//[1,512,1,1] -> [1,512,1,1]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_25_Conv2D", 2048, 1, 1, 0));//[1,512,1,1] -> [1,2048,1,1]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_25_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_6")); //[1,2048,1,1] -> [1,2048,1,1]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_21", CUDNN_ACTIVATION_RELU));//[1,2048,1,1] -> [1,2048,1,1]

    shortcut = mainline;
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_26_Conv2D", 512, 1, 1, 0));//[1,2048,1,1] -> [1,512,1,1]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_26_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_22", CUDNN_ACTIVATION_RELU));//[1,512,1,1] -> [1,512,1,1]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_27_Conv2D", 512, 3, 1, 1));//[1,512,1,1] -> [1,512,1,1]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_27_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_23", CUDNN_ACTIVATION_RELU));//[1,512,1,1] -> [1,512,1,1]
    mainline = model.add_layer(new Conv2D(mainline, "resnet_model_conv2d_28_Conv2D", 2048, 1, 1, 0));//[1,512,1,1] -> [1,2048,1,1]
    mainline = model.add_layer(new FusedBatchNormalization(mainline, "resnet_model_batch_normalization_28_FusedBatchNormV3",CUDNN_BATCHNORM_SPATIAL));

    mainline = model.add_layer(new Add(mainline, shortcut, "resnet_model_add_7")); //[1,2048,1,1] -> [1,2048,1,1]
    mainline = model.add_layer(new Activation(mainline, "resnet_model_Relu_24", CUDNN_ACTIVATION_RELU));//[1,2048,1,1] -> [1,2048,1,1]
    // block4 end

    mainline = model.add_layer(new Dense(mainline, "resnet_model_dense1", 10)); //[1,500,1,1] -> [1,10,1,1]
    mainline = model.add_layer(new Softmax(mainline, "softmax"));//[1,10,1,1] -> [1,10,1,1]

    model.cuda();

    if (load_pretrain)
        model.load_pretrain();
    model.train();

    // step 3. train
    int step = 0;
    Blob<float> *train_data = train_data_loader.get_data();
    Blob<float> *train_target = train_data_loader.get_target();
    train_data_loader.get_batch();
    int tp_count = 0;
    while (step < num_steps_train) {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
        train_data->to(cuda);
        train_target->to(cuda);

        // forward
        model.forward(train_data);
        tp_count += model.get_accuracy(train_target);
        // back-propagation
        model.backward(train_target);
        // update parameter
        // we will use learning rate decay to the learning rate
        //learning_rate *= 1.f / (1.f + lr_decay * step);
        //model.update(learning_rate);
        model.update_adam(learning_rate, beta1, beta2, eps_hat, step);
        // fetch next data
        step = train_data_loader.next();

        // nvtx profiling end
        nvtxRangePop();

        // calculation softmax loss
        if (step % monitoring_step == 0) {
            float loss = model.loss(train_target);
            float accuracy = 100.f * tp_count / monitoring_step / batch_size_train;

            std::cout << "step: " << std::right << std::setw(4) << step << ", loss: " << std::left << std::setw(5)
                      << std::fixed << std::setprecision(3) << loss << ", accuracy: " << accuracy << "%" << std::endl;

            tp_count = 0;
        }
    }

    // trained parameter save
    if (file_save)
        model.write_file();

    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    // model.train();
    model.test();

    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < num_steps_test) {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());
        // update shared buffer contents
        test_data->to(cuda);
        test_target->to(cuda);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
    }

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;

    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}
