# Compilador
CXX := clang++

# Opciones de compilación
CXXFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

# Archivos fuente
SOURCES := main.dp.cpp dwt.dp.cpp components.dp.cpp dwt_cuda/fdwt53.dp.cpp dwt_cuda/fdwt97.dp.cpp dwt_cuda/rdwt97.dp.cpp dwt_cuda/rdwt53.dp.cpp

# Convertir los archivos fuente en nombres de archivos objeto
OBJECTS := $(SOURCES:.dp.cpp=.o)

# Nombre del ejecutable
EXECUTABLE := dwt2d

# Regla por defecto
all: $(EXECUTABLE)

# Regla para compilar cada archivo fuente a un archivo objeto
%.o: %.dp.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para enlazar los archivos objeto en el ejecutable final
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f $(OBJECTS) $(EXECUTABLE) *.o