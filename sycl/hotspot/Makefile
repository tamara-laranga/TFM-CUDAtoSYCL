
CC := clang++
FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SRC = hotspot.dp.cpp

EXE = hotspot

release: $(SRC)
	$(CC) $(FLAGS) $(SRC) -o $(EXE) 

enum: $(SRC)
	$(CC) $(FLAGS) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(CC) $(FLAGS) -g $(SRC) -o $(EXE) 

debugenum: $(SRC)
	$(CC) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
