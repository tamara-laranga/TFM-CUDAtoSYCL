CC := clang++

CCFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda 
SRC = 3D.dp.cpp

EXE = 3D 

OUTPUT = *.out


release: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) -o $(EXE) 

enum: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) -deviceemu  -o $(EXE) 

debug: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) -g  -o $(EXE) 

debugenum: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) -g -deviceemu  -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
