

CC := clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SRC = srad.dp.cpp

EXE = srad

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) 

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
