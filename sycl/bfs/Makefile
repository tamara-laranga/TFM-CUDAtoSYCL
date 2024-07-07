include ../../common/make.config

#CC := $(CUDA_DIR)/bin/nvcc
CC := clang++
CC_FLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SRC = bfs.dp.cpp

EXE = bfs.out

release: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -I../util -DTIMING

enum: $(SRC)
	$(CC) $(CC_FLAGS) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(CC)  $(CC_FLAGS) -g $(SRC) -o $(EXE)

debugenum: $(SRC)
	$(CC) $(CC_FLAGS) -g -deviceemu $(SRC) -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
