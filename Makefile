TARGET=main
OBJECTS=src/main.o src/util.o src/reduction.o
INCLUDES=-I/usr/local/cuda/include -I./include 

CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
CUDA_ARCH=70 # for Volta
LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm -lcublas -lnvToolsExt

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f $(INCLUDES)
CUDA_CFLAGS:=$(foreach opt, $(CPPFLAGS),-Xcompiler=$(opt)) -arch=sm_$(CUDA_ARCH) 


all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

src/%.o: src/%.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)