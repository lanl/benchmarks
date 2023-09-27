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


def osu_allgatherv(args):

    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    numprocs = comm.Get_size()

    options = Options("Allgatherv", args)
    util.check_numprocs(numprocs, myid, limit=3)
    util.print_header(options.benchmark, myid)

    structure = util.find_structure(options.buffer)

    r_buf = util.allocate(options.max_message_size*numprocs, structure)
    s_buf = util.allocate(options.max_message_size, structure)

    def array_int(n): return array('i', [0]*n)
    r_counts = array_int(numprocs)
    r_displs = array_int(numprocs)

    for size in util.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = list(range(options.iterations+options.skip))
        disp = 0
        for i in range(numprocs):
            r_counts[i] = size
            r_displs[i] = disp
            disp += size

        s_msg = [s_buf, size, MPI.BYTE]
        r_msg = [r_buf, r_counts, r_displs, MPI.BYTE]

        comm.Barrier()
        for i in iterations:
            if i == options.skip:
                t_start = MPI.Wtime()
            comm.Allgatherv(s_msg, r_msg)
        t_end = MPI.Wtime()
        comm.Barrier()

        util.print_stats(t_end, t_start, options.iterations,
                         myid, comm, numprocs, size)
