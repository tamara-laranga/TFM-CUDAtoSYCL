CXX=clang++
CXX_OPTS=-O3 -m64 -g -fsycl -fsycl-targets=nvptx64-nvidia-cuda

OBJ = main_test_cu.dp.o stats_logger.o vlc_kernel_sm64huff.dp.o scan.dp.o pack_kernels.dp.o cpuencode.o
EXE = pavle

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

all:  $(EXE)

$(EXE): $(OBJ) 
	$(CXX) $(CXX_OPTS) $(OBJ) -o $(EXE)

main_test_cu.dp.o: main_test_cu.dp.cpp 
	$(CXX) $(TESTING) $(CACHECWLUT) $(CXX_OPTS) -c main_test_cu.dp.cpp -o main_test_cu.dp.o

stats_logger.o: stats_logger.cpp
	$(CXX) $(CXX_OPTS) -c -o stats_logger.o stats_logger.cpp

vlc_kernel_sm64huff.dp.o: vlc_kernel_sm64huff.dp.cpp 
	$(CXX) $(TESTING) $(CXX_OPTS) -c vlc_kernel_sm64huff.dp.cpp 

scan.dp.o: scan.dp.cpp
	$(CXX) -c scan.dp.cpp $(CXX_OPTS)

pack_kernels.dp.o: pack_kernels.dp.cpp 
	$(CXX) -c pack_kernels.dp.cpp $(CXX_OPTS)

cpuencode.o: cpuencode.cpp
	$(CXX) $(CXX_OPTS) -c cpuencode.cpp 

clean:
	rm -f *.o $(EXE) 
