# Makefile for memory_recorder.

# EXPECT THESE TO BE SET IN ENV OR ON COMMAND LINE.
CXX		?= g++
FC      ?= gfortran
PREFIX	?= ${HOME}/.local

# DOESN'T WORK WITH OPTIMIZATION
# FLAGS, ADDITIONAL FLAGS CAN BE ADDED ON CL.
CXXFLAGS   += -O0 -g -fPIC -std=c++11
FCFLAGS    += -O0 -g -fPIC
LDFLAGS    += -shared -ldl
LIBFLAGS   += -L. -lmemory_recorder

# TARGET AND DEPS
TARGETS    = libmemory_recorder.so memrecord
CPPSOURCE  = memory_recorder.cpp memory_recorder.h Makefile
FCSOURCE   = memory_recorder.f memory_recorder.cpp Makefile
LIBSOURCE  = libmem_rec_c.o libmem_rec_f.o
TESTSRC    = tests/fortran_wrapper.f

# RULES
.PHONY: clean test

default: $(TARGETS)

libmem_rec_c.o: $(CPPSOURCE)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

libmem_rec_f.o: $(FCSOURCE)
	$(FC) -c $< -o $@ $(FCFLAGS)

libmemory_recorder.so: $(LIBSOURCE)
	$(CXX) $^ -o $@ $(LDFLAGS)

memrecord: main.cpp libmemory_recorder.so
	$(CXX) $< -o $@ $(LIBFLAGS) $(CXXFLAGS)

test: $(TESTSRC) libmemory_recorder.so
	$(FC) $< -o fortran_test $(LIBFLAGS) $(FCFLAGS) 

clean:
	rm -f $(TARGETS) *.o *_test *.csv *.out *.mod

install: $(TARGETS) utils/collect_results.py
	install -Dm755 "memrecord" $(PREFIX)/bin/memrecord
	install -Dm755 "utils/collect_results.py" $(PREFIX)/bin/mem_collect
	install -Dm755 "libmemory_recorder.so" $(PREFIX)/lib/libmemory_recorder.so
	install -Dm644 "memory_recorder.h" $(PREFIX)/include/memory_recorder.h
	@-mv -f MEMORY_RECORDER.mod memory_recorder.mod 2> /dev/null || true
	install -Dm644 "memory_recorder.mod" $(PREFIX)/include/memory_recorder.mod