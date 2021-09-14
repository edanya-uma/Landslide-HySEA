.SUFFIXES:	.cu

export RAIZ 	=/share/apps/nvidia/cuda-8.0
export NVCC	=$(RAIZ)/bin/nvcc
export CXX      = /share/apps/OPENMPI-2.1.2/bin/mpic++
export INC	=-I../ -I/share/apps/OPENMPI-2.1.2/include -I/share/apps/NETCDF_C/include -I$(RAIZ)/include
export LIB	=-L$(RAIZ)/lib64 -lcudart -L./GPU -lpnetcdf
export NVCCFLAGS =-arch=sm_60

OBJS	:= Arista_kernel.o Complex.o ComprobarSoporteCUDA.o Matriz.o netcdf.o Reduccion_kernel.o ShallowWater.o Volumen_kernel.o main.o Problema.o

%.o : %.cxx
	$(CXX) $(INC) -c $*.cxx

%.o : %.cu
	$(NVCC) $(INC) $(LIB) $(NVCCFLAGS) -c $*.cu
	
all: lib2D_AVALANCHAS_MGPU_NETCDF.a

lib2D_AVALANCHAS_MGPU_NETCDF.a : $(OBJS)
	$(NVCC) --lib $(OBJS) -o lib2D_AVALANCHAS_MGPU_NETCDF.a

.PHONY: clean
clean:
	rm -fr *.o *~
	rm lib2D_AVALANCHAS_MGPU_NETCDF.a