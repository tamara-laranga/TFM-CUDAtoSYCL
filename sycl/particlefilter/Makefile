#makefile

CC := clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda

all: naive float

naive: ex_particle_CUDA_naive_seq.dp.cpp
	$(CC) -g -lm -O3 -use_fast_math  ex_particle_CUDA_naive_seq.dp.cpp -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.dp.cpp
	$(CC) -g -lm -O3 -use_fast_math ex_particle_CUDA_float_seq.dp.cpp -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float
