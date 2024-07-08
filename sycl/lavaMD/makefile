
# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

CXX= clang++
# OMP_FLAG = 	-Xcompiler paste_one_here
CC_FLAG = -fsycl -fsycl-targets=nvptx64-nvidia-cuda

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu_cuda_wrapper.dp.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.dp.o
	$(CXX)	$(CC_FLAG) main.o \
			./kernel/kernel_gpu_cuda_wrapper.dp.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.dp.o \
			-lm \
			$(OMP_LIB) \
			-o lavaMD

# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			./kernel/kernel_gpu_cuda_wrapper.h \
			./kernel/kernel_gpu_cuda_wrapper.dp.cpp \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c \
			./util/device/device.h \
			./util/device/device.dp.cpp
	$(CXX)	$(CC_FLAG) $(OUTPUT) main.c \
			-c \
			-o main.o \
			-O3

./kernel/kernel_gpu_cuda_wrapper.dp.o:	./kernel/kernel_gpu_cuda_wrapper.h \
									./kernel/kernel_gpu_cuda_wrapper.dp.cpp
	$(CXX) 	$(CC_FLAG)				./kernel/kernel_gpu_cuda_wrapper.dp.cpp \
									-c \
									-o ./kernel/kernel_gpu_cuda_wrapper.dp.o \
									-O3 

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

./util/device/device.dp.o:	./util/device/device.h \
						./util/device/device.dp.cpp
	$(CXX)	$(CC_FLAG)	./util/device/device.dp.cpp \
						-c \
						-o ./util/device/device.dp.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
		./util/device/*.o \
		lavaMD
