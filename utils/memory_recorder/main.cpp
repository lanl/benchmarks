#include "memory_recorder.h"

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    int rank, size;
    const unsigned long mb = 1024*1024;

    // Default 256 MiB per array per process.
    long long array_size = mb*64;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MemoryRecorder mem_record = MemoryRecorder();

    if (argc > 1) {
        array_size = atoll(argv[1]);
    }

    // Allocate
    float *s1, *s2, *s3;
    long long array_mem = array_size*sizeof(float);
    s1 = (float*)malloc(array_mem);
    s2 = (float*)malloc(array_mem);
    s3 = (float*)malloc(array_mem);

    // Pause and read meminfo
    MPI_Barrier(MPI_COMM_WORLD);
    mem_record.read_meminfo("PostMalloc");

    // Fill arrays with time and add them.
    for (long k=0; k<array_size; k++){
        s1[k] = time(0);
        s2[k] = time(0);
        s3[k] = s2[k] + s1[k];
    }

    // Pause and read meminfo
    MPI_Barrier(MPI_COMM_WORLD);
    mem_record.read_meminfo("PostFill");

    // Free
    free(s1);
    free(s2);
    free(s3);

    // Pause and read meminfo
    MPI_Barrier(MPI_COMM_WORLD);
    mem_record.read_meminfo("PostFree");

    // Write out the meminfo and RSSMax info
    mem_record.write_rss();
    mem_record.write_meminfo();

    // Tell how much memory was allocated in the mallocs in main.
    if (rank == 0) {
        std::cout << "Total HEAP Allocated Per Proc: " << 3*array_mem/mb << " (MiB)" << std::endl;
    }

    // Finalize the MPI environment
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}