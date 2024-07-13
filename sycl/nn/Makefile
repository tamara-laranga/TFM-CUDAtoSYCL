
LOCAL_CC = -g -O3 -Wall
CC := clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.dp.cpp
	$(CC) $(LOCAL_CC) -o nn nn_cuda.dp.cpp -I../util -DTIMING

clang: $(SRC)
	clang++ nn_cuda.dp.cpp -o nn -I../util \
	 -ldl -lrt -pthread -DTIMING

hurricane_gen : hurricane_gen.c
	$(CC) $(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
