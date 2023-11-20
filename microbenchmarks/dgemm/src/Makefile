
include build_config/$(ARCHTYPE).gm

mt-dgemm: mt-dgemm.c
	$(CC) $(CFLAGS) -o mt-dgemm mt-dgemm.c ${LDFLAGS}

clean:
	rm mt-dgemm *.o

