.SUFFIXES:	.cu .c

export UNAME = $(shell uname -m)

# MODIFY ONLY HERE
export ROOT             = /share/apps/nvidia/cuda-8.0
export OPENMPI          = /share/apps/OPENMPI-2.1.2

# Compilers and Libraries
export NVCC     = $(ROOT)/bin/nvcc
export CXX              = $(OPENMPI)/bin/mpic++
export CXX_LIB                 = $(OPENMPI)/lib
export XX_INCLUDE             = $(OPENMPI)/include

export CXXFLAGS         = -O3 -DNDEBUG
export CXXINC           = -I./ -I$(ROOT)/include -I$(CXX_INCLUDE)
export CXXLIBS          =-L$(CXX_LIB) -L./GPU -l2D_AVALANCHAS_MGPU_NETCDF -L$(ROOT)/lib64 -lcudart -lpnetcdf

OBJSC   := cond_ini.o

%.o: 	%.cxx
	$(CXX) $(CXXINC) -c $*.cxx

all: L-HySEA.exe

tsunami.exe: $(OBJSC)
	$(CXX) $(CXXFLAGS) -o $@ $(CXXLIBS) $(CXXINC)

.PHONY: clean
clean:
	rm -rf *.o *~ *.exe
        