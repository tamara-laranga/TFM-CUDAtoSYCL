# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
DPCPP = clang++
DPCPP_FLAGS =  -fsycl -fsycl-targets=nvptx64-nvidia-cuda

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	DPCPP_FLAGS += -g -O0
else
	DPCPP_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	DPCPP_FLAGS += -deviceemu
endif


backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(DPCPP) $(DPCPP_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.dp.cpp backprop.h
	$(DPCPP) $(DPCPP_FLAGS) -c backprop_cuda.dp.cpp -o backprop_cuda.o

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
