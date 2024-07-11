CC := clang++

CC_FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort: main.dp.cpp  bucketsort.dp.cpp bucketsort.dp.hpp bucketsort_kernel.dp.cpp histogram1024_kernel.dp.cpp  mergesort.dp.cpp mergesort.dp.hpp mergesort_kernel.dp.cpp
	$(CC) $(CC_FLAGS) $(OUTPUT) $(TIMER) bucketsort.dp.cpp mergesort.dp.cpp main.dp.cpp -o hybridsort


clean:
	rm	*.o hybridsort
