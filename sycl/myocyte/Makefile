# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

CC_FLAG = -fsycl -fsycl-targets=nvptx64-nvidia-cuda

# link objects(binaries) together
myocyte.out:		main.o
	clang++	main.dp.o \
				 \
				-lm  $(CC_FLAG) \
                                -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				\
				main.dp.cpp \
				work.dp.cpp \
				solver.dp.cpp \
				embedded_fehlberg_7_8.dp.cpp \
				master.dp.cpp \
				kernel.dp.cpp \
				kernel_ecc.dp.cpp \
				kernel_cam.dp.cpp \
				kernel_fin.dp.cpp \
				work_2.dp.cpp \
				solver_2.dp.cpp \
				embedded_fehlberg_7_8_2.dp.cpp \
				kernel_2.dp.cpp \
				kernel_ecc_2.dp.cpp \
				kernel_cam_2.dp.cpp \
				kernel_fin_2.dp.cpp \
				file.c \
				timer.c
	clang++	 main.dp.cpp $(CC_FLAG)\
				-c -O3 -g 

# delete all object files
clean:
	rm *.o *.out output.txt
