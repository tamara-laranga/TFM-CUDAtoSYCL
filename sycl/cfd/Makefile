CC_FLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.dp.cpp
	clang++ $(CC_FLAGS) -O2 -v euler3d.dp.cpp -o euler3d -I inc 

euler3d_double: euler3d_double.dp.cpp
	clang++ $(CC_FLAGS) -v -O3 euler3d_double.dp.cpp -o euler3d_double -I inc 

pre_euler3d: pre_euler3d.dp.cpp
	clang++ $(CC_FLAGS) -v -O3 pre_euler3d.dp.cpp -o pre_euler3d -I inc 

pre_euler3d_double: pre_euler3d_double.dp.cpp
	clang++ $(CC_FLAGS) -v -O3 pre_euler3d_double.dp.cpp -o pre_euler3d_double -I inc 


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
