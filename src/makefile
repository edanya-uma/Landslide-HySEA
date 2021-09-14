.SUFFIXES:	.cu .c

export UNAME = $(shell uname -m)
export RAIZ = /share/apps/nvidia/cuda-8.0

# Compilers
export NVCC 	= $(RAIZ)/bin/nvcc
export CXX 		= /share/apps/OPENMPI-2.1.2/bin/mpic++

export CXXFLAGS		= -O3 -DNDEBUG
export CXXINC 		= -I./ -I$(RAIZ)/include -I/share/apps/OPENMPI-2.1.2/include

export CXXLIBS		=-L/share/apps/OPENMPI-2.1.2/lib -L./GPU -l2D_AVALANCHAS_MGPU_NETCDF -L$(RAIZ)/lib64 -lcudart -lpnetcdf

OBJSC 	:= cond_ini.o

%.o: 	%.cxx
	$(CXX) $(CXXINC) -c $*.cxx

all: tsunami.exe

tsunami.exe: $(OBJSC)
	$(CXX) $(CXXFLAGS) -o $@ $(CXXLIBS) $(CXXINC)

.PHONY: clean
clean:
	rm -rf *.o *~ *.exe
        