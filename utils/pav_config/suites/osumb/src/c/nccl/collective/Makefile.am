AUTOMAKE_OPTIONS = subdir-objects

NVCC = nvcc
NVCFLAGS = -cuda -maxrregcount 32 -ccbin $(CXX) $(NVCCFLAGS)
SUFFIXES = .cu .cpp
.cu.cpp:
	$(NVCC) $(NVCFLAGS) $(INCLUDES) $(CPPFLAGS) --output-file $@.ii $<
	mv $@.ii $@

ncclcollectivedir = $(pkglibexecdir)/c/nccl/collective
ncclcollective_PROGRAMS = osu_nccl_allgather osu_nccl_allreduce osu_nccl_bcast osu_nccl_reduce osu_nccl_reduce_scatter osu_nccl_alltoall

AM_CFLAGS = -I${top_srcdir}/c/util

UTILITIES = ../../util/osu_util.c ../../util/osu_util.h ../../util/osu_util_mpi.c ../../util/osu_util_mpi.h ../../util/osu_util_nccl.c ../../util/osu_util_nccl.h
if CUDA_KERNELS
UTILITIES += ../../util/kernel.cu
if BUILD_USE_PGI
AM_CXXFLAGS = --nvcchost --no_preincludes
endif
endif

osu_nccl_allgather_SOURCES = osu_nccl_allgather.c $(UTILITIES)
osu_nccl_allreduce_SOURCES = osu_nccl_allreduce.c $(UTILITIES)
osu_nccl_bcast_SOURCES = osu_nccl_bcast.c $(UTILITIES)
osu_nccl_reduce_SOURCES = osu_nccl_reduce.c $(UTILITIES)
osu_nccl_reduce_scatter_SOURCES = osu_nccl_reduce_scatter.c $(UTILITIES)
osu_nccl_alltoall_SOURCES = osu_nccl_alltoall.c $(UTILITIES)

if EMBEDDED_BUILD
    AM_LDFLAGS =
    AM_CPPFLAGS = -I$(top_builddir)/../src/include \
          -I${top_srcdir}/util \
		  -I${top_srcdir}/../src/include
if BUILD_PROFILING_LIB
    AM_LDFLAGS += $(top_builddir)/../lib/lib@PMPILIBNAME@.la
endif
    AM_LDFLAGS += $(top_builddir)/../lib/lib@MPILIBNAME@.la
endif

if OPENACC
    AM_CFLAGS += -acc -ta=tesla:nordc
    AM_CXXFLAGS = -acc -ta=tesla:nordc
endif
