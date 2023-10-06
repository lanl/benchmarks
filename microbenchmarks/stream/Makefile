#
MPICC ?= $(PAV_MPICC)
CC ?= $(PAV_CC)
CFLAGS ?= $(PAV_CFLAGS)

FF ?= $(PAV_FC)
FFLAGS ?= $(PAV_FFLAGS)

ifeq ($(MPICC),)
	MPICC=$(CC)
endif

all: xrds-stream.exe stream_f.exe stream_c.exe stream_mpi.exe

stream_f.exe: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FF) $(FFLAGS) -c stream.f
	$(FF) $(FFLAGS) stream.o mysecond.o -o stream_f.exe

clean:
	rm -f *stream*.exe *.o

stream_c.exe: stream.c
	$(CC) $(CFLAGS) stream.c -o stream_c.exe

stream_mpi.exe: stream_mpi.c
	$(MPICC) $(CFLAGS) stream_mpi.c -o stream_mpi.exe

xrds-stream.exe: xrds-stream.c
	$(CC) $(CFLAGS) xrds-stream.c -o xrds-stream.exe
