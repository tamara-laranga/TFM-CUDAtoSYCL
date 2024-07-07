
CC := clang++
CC_FLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SRC = gaussian.dp.cpp
EXE = gaussian

release: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) 

clang: $(SRC)
	clang++ $(SRC) $(CC_FLAGS) -o $(EXE) -I../util  \
	 -ldl -lrt -pthread -DTIMING

clean:
	rm gaussian
