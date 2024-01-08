# Makefile for memory_recorder.

# EXPECT THESE TO BE SET IN ENV OR ON COMMAND LINE.
CXX		?= g++
PREFIX	?= ${HOME}/.local

# DOESN'T WORK WITH OPTIMIZATION
# FLAGS, ADDITIONAL FLAGS CAN BE ADDED ON CL.
CXXFLAGS   += -O0 -g -fPIC -std=c++11
LDFLAGS    += -shared
LIBFLAGS   += -L. -lmemory_recorder

# TARGET AND DEPS
TARGETS    = libmemory_recorder.so memrecord
LIBSOURCE  = memory_recorder.cpp memory_recorder.h Makefile

# RULES
.PHONY: clean

default: $(TARGETS)

libmemory_recorder.so: $(LIBSOURCE)
	$(CXX) $< -o $@ $(CXXFLAGS) $(LDFLAGS)

memrecord: main.cpp libmemory_recorder.so
	$(CXX) $< -o $@ $(LIBFLAGS) $(CXXFLAGS)

clean:
	rm -f $(TARGETS)

install: $(TARGETS) utils/collect_results.py
	install -Dm755 "memrecord" $(PREFIX)/bin/memrecord
	install -Dm755 "utils/collect_results.py" $(PREFIX)/bin/mem_collect
	install -Dm755 "libmemory_recorder.so" $(PREFIX)/lib/libmemory_recorder.so
	install -Dm644 "memory_recorder.h" $(PREFIX)/include/memory_recorder.h