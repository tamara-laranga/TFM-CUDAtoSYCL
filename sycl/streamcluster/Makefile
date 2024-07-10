CC = clang++
CC_FLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
TARGET_G = sc_gpu # Reemplaza esto con el nombre de tu archivo de salida

# Lista de archivos fuente
SOURCES = streamcluster_cuda_cpu.cpp.dp.cpp streamcluster_cuda.dp.cpp streamcluster_header.dp.cpp

# Regla por defecto
all: $(TARGET_G)

$(TARGET_G): $(SOURCES)
	$(CC) $(CC_FLAGS) $(SOURCES) -o $(TARGET_G)

clean:
	rm -f $(TARGET_G)