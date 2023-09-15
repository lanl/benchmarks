
CC=gcc
CFLAGS=-ffast-math -mavx2 -ftree-vectorizer-verbose=3 -O3 -fopenmp -DUSE_CBLAS
LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

mt-dgemm: mt-dgemm.c
	$(CC) $(CFLAGS) -o mt-dgemm mt-dgemm.c

clean:
	rm mt-dgemm *.o

