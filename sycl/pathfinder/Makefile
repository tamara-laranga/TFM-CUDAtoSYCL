CC := clang++
FLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SRC = pathfinder.dp.cpp

EXE = pathfinder

release:
	$(CC) $(FLAGS) $(SRC) -o $(EXE) 

clang: $(SRC)
	clang++ $(SRC) $(FLAGS) -o $(EXE) -I../util  \
		-ldl -lrt -pthread -DTIMING

enum:
	$(CC) $(FLAGS) -deviceemu $(SRC) -o $(EXE) 

debug:
	$(CC) $(FLAGS) -g $(SRC) -o $(EXE) 

debugenum:
	$(CC) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) 

clean:
	rm -f pathfinder
