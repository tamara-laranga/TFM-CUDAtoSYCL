
CXX = clang++

DEFS += \
		-DGPU_TIMER 
CXXFLAGS += -I../common \
			 -O3 \
			 -use_fast_math \
			 -Wall \
			 -lm \
			 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.dp.cpp
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c $<

%.o:	%.dp.cpp 
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c $<

clang: $(SRC)
	clang++ lud.dp.cpp lud_kernel.dp.cpp ../common/common.c -o $(EXECUTABLE) \
		-I../common -I../../utilS \
		-ldl -lrt -pthread -DTIMING

$(EXECUTABLE) : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@  $?

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
