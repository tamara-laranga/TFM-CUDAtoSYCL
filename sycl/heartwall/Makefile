
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

CXX = clang++
FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda

# link objects(binaries) together
heartwall: main.dp.o ./AVI/avilib.o ./AVI/avimod.o 
	$(CXX) $(FLAGS) main.dp.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall

# compile main function file into object (binary)
main.dp.o: main.dp.cpp kernel.dp.cpp define.c
	$(CXX) $(FLAGS) $(OUTPUT) main.dp.cpp -I./AVI -c -O3

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
