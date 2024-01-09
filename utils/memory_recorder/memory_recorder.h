#ifndef memory_recorder_h_
#define memory_recorder_h_

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <sstream>
#include <map>
#include <numeric>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <unistd.h>

#include <mpi.h>

// TODO, 
// * Option to output continuously or capture on failure?

double round_pct(double num_in) {
    double round1 = std::round(num_in * 10000);
    return round1 / 100.0;
}

struct MemoryRecorder
{
    const unsigned long kb = 1024;
    const unsigned long mb = 1024*1024;

    int getrss_summary, getmeminfo, pid, numnodes;
    int globalrank, localrank, localsize, globalsize, nodenum;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm shmcomm;

    long *rss_collect;
    long local_maxrss, global_maxrss;

    std::string memfileOut, mempctfileOut, rssfileOut, line, word;
    std::map<std::string, std::vector<long>> freemem;
    std::map<std::string, std::vector<double>> freemem_pct;
    std::vector<double> totalnumamem;
    std::vector<std::string> meminfo_names;

    MemoryRecorder();
    static unsigned long getMaxRSS();
    static long getRamSize();
    void summarizeMaxRSS();
    void read_meminfo(std::string const &loc);
    void write_meminfo();
    void write_rss();
    ~MemoryRecorder();
};


#endif


