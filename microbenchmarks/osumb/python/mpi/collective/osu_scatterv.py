"""
Copyright (C) 2002-2022 the Network-Based Computing Laboratory
(NBCL), The Ohio State University.

Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)

For detailed copyright and licensing information, please refer to the
copyright file COPYRIGHT in the top level OMB directory.
"""

from mpi4py import MPI
from array import array
from util.osu_util_mpi import util
from util.options import Options

def osu_scatterv(args):

    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    numprocs = comm.Get_size()

    options = Options("Scatterv", args)
    util.check_numprocs(numprocs, myid, limit=3)
    util.print_header(options.benchmark, myid)

    structure = util.find_structure(options.buffer)
    s_buf = util.allocate(options.max_message_size*numprocs, structure)
    r_buf = util.allocate(options.max_message_size, structure)

    def array_int(n): return array('i', [0]*n)
    s_counts = array_int(numprocs)
    s_displs = array_int(numprocs)

    for size in util.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = list(range(options.iterations+options.skip))
        disp = 0
        for i in range(numprocs):
            s_counts[i] = size
            s_displs[i] = disp
            disp += size

        r_msg = [r_buf, size, MPI.BYTE]
        if myid == 0:
            s_msg = [s_buf, s_counts, s_displs, MPI.BYTE]
        else:
            s_msg = None
            r_msg = [r_buf, size, MPI.BYTE]

        if options.pickle:
            s_msg = s_buf[0:size]
            comm.Barrier()
            for i in iterations:
                if i == options.skip:
                    t_start = MPI.Wtime()
                r_msg = comm.scatter([(s_msg) for i in range(numprocs)], 0)
            t_end = MPI.Wtime()
            comm.Barrier()
        else:
            comm.Barrier()
            for i in iterations:
                if i == options.skip:
                    t_start = MPI.Wtime()
                comm.Scatterv(s_msg, r_msg, 0)
            t_end = MPI.Wtime()
            comm.Barrier()

        util.print_stats(t_end, t_start, options.iterations,
                         myid, comm, numprocs, size)
