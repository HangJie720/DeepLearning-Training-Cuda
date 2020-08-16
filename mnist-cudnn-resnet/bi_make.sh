clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda -Ieigen \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc \
-lcudart -lcublas -lcudnn -lgomp -lcurand -lnvToolsExt \
-c train.cpp -o obj/train.o

clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda -Ieigen \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc \
-lcudart -lcublas -lcudnn -lgomp -lcurand -lnvToolsExt \
-c src/mnist.cpp -o obj/mnist.o

clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda -Ieigen \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc \
-lcudart -lcublas -lcudnn -lgomp -lcurand -lnvToolsExt \
-c src/loss.cu -o obj/loss.o

clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda -Ieigen \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc \
-lcudart -lcublas -lcudnn -lgomp -lcurand -lnvToolsExt \
-c src/layer.cu -o obj/layer.o

clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda -Ieigen \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc \
-lcudart -lcublas -lcudnn -lgomp -lcurand -lnvToolsExt \
-c src/network.cpp -o obj/network.o


clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda -Ieigen \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc \
-lcudart -lcublas -lcudnn -lgomp -lcurand -lnvToolsExt \
-o train_bi obj/train.o obj/mnist.o obj/loss.o obj/layer.o obj/network.o

