
CC := clang++ 
FLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SRC = needle.dp.cpp

EXE = needle

release: $(SRC)
	$(CC) $(FLAGS) $(SRC) -o $(EXE) 

clang: $(SRC)
	clang++ $(FLAGS) $(SRC) -o $(EXE) -I../util  \
		-ldl -lrt -pthread -DTIMING

enum: $(SRC)
	$(CC) $(FLAGS) -deviceemu $(SRC) -o $(EXE)

debug: $(SRC)
	$(CC) $(FLAGS) -g $(SRC) -o $(EXE) 

debugenum: $(SRC)
	$(CC) $(FLAGS) ${KERNEL_DIM} -g -deviceemu $(SRC) -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
