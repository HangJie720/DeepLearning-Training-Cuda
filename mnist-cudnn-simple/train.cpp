#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvToolsExt.h>

#define SWITCH_CHAR             '-'
using namespace cudl;

int main(int argc, char *argv[]) {
    /* configure the network */
    int batch_size_train = 256;
    int num_steps_train = 2400;
    int monitoring_step = 200;

    double learning_rate = 0.04f;
    double lr_decay = 0.00005f;

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
    //model.add_layer(new Pad("pad1", {0, 0, 0, 0, 3, 3, 3, 3}, 0)); //[1,1,28,28] -> [1,1,32,32]
    model.add_layer(new Conv2D("conv1", 32, 5, 1, 2)); //[1,1,32,32] -> [1,32,28,28]
    model.add_layer(new FusedBatchNormalization("fbn1", CUDNN_BATCHNORM_SPATIAL)); //[1,1,28,28] -> [1,32,28,28]
    model.add_layer(new Activation("relu1", CUDNN_ACTIVATION_RELU)); //[1,1,28,28] -> [1,32,28,28]
    model.add_layer(new Pooling("pool1", 2, 0, 2, CUDNN_POOLING_MAX)); //[1,1,28,28] -> [1,32,14,14]

    model.add_layer(new Conv2D("conv2", 64, 5, 1, 2)); //[1,32,14,14] -> [1,64,14,14]
    model.add_layer(new FusedBatchNormalization("fbn2", CUDNN_BATCHNORM_SPATIAL)); //[1,64,14,14] -> [1,64,14,14]
    model.add_layer(new Activation("relu2", CUDNN_ACTIVATION_RELU)); //[1,64,14,14] -> [1,64,14,14]
    model.add_layer(new Pooling("pool2", 2, 0, 2, CUDNN_POOLING_MAX)); //[1,64,14,14] -> [1,64,7,7]

    model.add_layer(new Dense("dense1", 1024)); //[1,64,7,7] -> [1,1024,1,1]
    model.add_layer(new Activation("relu3", CUDNN_ACTIVATION_RELU)); //[1,1024,1,1] -> [1,1024,1,1]
    model.add_layer(new Dense("dense2", 10)); //[1,1024,1,1]->[1,10,1,1]
    model.add_layer(new Softmax("softmax")); //[1,10,1,1]->[1,10,1,1]
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
        learning_rate *= 1.f / (1.f + lr_decay * step);
        model.update(learning_rate);
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

    //step 2. model initialization
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
