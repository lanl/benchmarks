#include "memory_recorder.h"


MemoryRecorder::MemoryRecorder() {

    int k;
    struct stat st = {0};
    local_maxrss.val=0;
    global_maxrss=0;
    getrss_summary=0;
    getmeminfo=0;

    MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);
    MPI_Comm_size(MPI_COMM_WORLD, &globalsize);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
        MPI_INFO_NULL, &eachnode);

    MPI_Comm_rank(eachnode, &localrank);
    MPI_Comm_size(eachnode, &localsize);
    MPI_Get_processor_name(hostname, &namelen);
    MPI_Comm_split(MPI_COMM_WORLD, localrank, globalrank, &bosscomm);
    MPI_Comm_rank(bosscomm, &bossrank);
    MPI_Comm_size(bosscomm, &bosssize);

    local_maxrss.index = bossrank;
    nodenum = globalrank/localsize;
    numnodes = globalsize/localsize;
    pid = getpid();

    rss_collect = (long*)malloc(sizeof(long) * localsize);

    // Make the hostname a string and add relative host number.
    std::string strhost, strhostnum;
    for (k=0; k<namelen; k++) {
        if (hostname[k] == '.') break;
        strhost += hostname[k];
    }

    this->gethostlist();

    // Get meminfo files to read from.
    if (localrank == 0) {
        k=0;
        strhostnum = std::to_string(nodenum) + "_" + strhost;
        char fname_in[128];
        snprintf(fname_in, sizeof(fname_in),
            "/sys/devices/system/node/node%d/meminfo",k);

        while (stat(fname_in, &st) == 0) {
            meminfo_names.push_back(fname_in);
            k++;
            snprintf(fname_in, sizeof(fname_in),
                "/sys/devices/system/node/node%d/meminfo",k);
        }

        // Set paths
        memfileOut = "meminfo_" + strhostnum + ".csv";
        mempctfileOut = "pctmeminfo_" + strhostnum + ".csv";
        rssfileOut = "rss_" + strhostnum + ".memout";
    }
}

long MemoryRecorder::getRamSize() {
    struct sysinfo infosys;
    sysinfo(&infosys);
    return infosys.totalram; // in bytes
}

unsigned long MemoryRecorder::getMaxRSS() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    unsigned long maxrss_func = usage.ru_maxrss;
    return maxrss_func; // In kb
}

void MemoryRecorder::summarizeMaxRSS() {

    int i;
    unsigned long maxrss = this->getMaxRSS();
    getrss_summary = 1;

    // Sum all Max Rss to get global, sum maxrss on node comm to get node MaxRss
    // Collect all Max Rss to print each out individually.
    MPI_Reduce(&maxrss, &global_maxrss, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(&maxrss, 1, MPI_LONG, rss_collect, 1, MPI_LONG, 0, eachnode);
    // MPI_Reduce(&maxrss, &local_maxrss, 1, MPI_LONG, MPI_SUM, 0, eachnode);

    if (localrank == 0) {
        for (i=0; i<localsize; i++) {
            local_maxrss.val += rss_collect[i];
        }
    }

    MPI_Reduce(&local_maxrss, &min_maxrss, 1, MPI_LONG_INT, MPI_MINLOC, 0, bosscomm);
    MPI_Reduce(&local_maxrss, &max_maxrss, 1, MPI_LONG_INT, MPI_MAXLOC, 0, bosscomm);
}

void MemoryRecorder::gethostlist() {
    std::string strhost;
    char hostnames[bosssize][MPI_MAX_PROCESSOR_NAME];
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostnames,
        MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, bosscomm); 
    
    if (globalrank == 0) {
        for (int k=0; k<bosssize; k++) {
            for (int b=0; b<namelen; b++) {
                if (hostnames[k][b] == '.') break;
                strhost+=hostnames[k][b];
            }
            hostlist.push_back(strhost);
            strhost="";
        }
    }
}

void MemoryRecorder::read_meminfo(std::string const &loc) {
    MPI_Barrier(eachnode);

    if (localrank == 0) {
        int i;
        std::vector<long> memnow;
        std::vector<double> mempctnow;
        std::ifstream stream;

        for (std::string fn : meminfo_names) {
            stream.open(fn, std::ios::in);
            std::getline(stream, line);

            // On the first time around, get the available memory on each numa node.
            if ( getmeminfo == 0 ) {
#ifdef MEMDEBUG
                if ( globalrank == 0 ) {
                    std::cout << "TOTALMEM " << fn << ": ";
                }
#endif
                std::stringstream line_stream(line);
                for (int i = 0; i < 4; i++) {
                    line_stream >> word;
#ifdef MEMDEBUG
                    if ( globalrank == 0 ) {
                        std::cout << word << ", ";
                    }
#endif
                }
#ifdef MEMDEBUG
                if ( globalrank == 0 ) {
                    std::cout << std::endl;
                }
#endif
                totalnumamem.push_back(std::stod(word));
            }

#ifdef MEMDEBUG
            // To make sure we're getting the right line, print the global rank 0 readings.
            if (getmeminfo == 0 && globalrank == 0 ) {
                std::cout << "Meminfo " << fn << ": " << line << std::endl;
            }
#endif

            // Extract the fourth field from the second line
            for (i = 0; i < 4; i++) {
                stream >> word; // Reads one space separated word from the stream.
#ifdef MEMDEBUG
                if (getmeminfo == 0 && globalrank == 0 ) {
                    std::cout << word << " ";
                }
#endif
            }

            memnow.push_back(std::stol(word));
            stream.close();
        }
        if ( getmeminfo == 0 ) {
            double totalmemsum = std::accumulate(totalnumamem.begin(), totalnumamem.end(), 0);
            totalnumamem.push_back(totalmemsum);
        }
        long memsum = std::accumulate(memnow.begin(), memnow.end(), 0);
        memnow.push_back(memsum);
        for (i=0; i<memnow.size(); i++) {
            mempctnow.push_back( round_pct( memnow.at(i)/totalnumamem.at(i) ) );
        }
        freemem.insert({loc, memnow});
        freemem_pct.insert({loc, mempctnow});
    }
    getmeminfo++;
    MPI_Barrier(eachnode);
}

void MemoryRecorder::write_meminfo() {

    // If you haven't recorded meminfo, don't write it out.
    if ( getmeminfo == 0 ) {
        if (globalrank == 0) {
            std::cout << "No Meminfo readings to write out " << std::endl;
        }
        return;
    }

    int i;
    double freem_pct;

    if (localrank == 0) {

        // Store the result in a csv file with a header.
        std::ofstream outfile;
        outfile.open(memfileOut, std::ios::out);
        outfile << "code_location,";
        for (i=0; i<meminfo_names.size(); i++) {
            outfile << i << ",";
        }
        outfile << "Total" << std::endl;
        for (const auto& item : freemem) {
            outfile << item.first << ",";
            outfile << std::setprecision(4);
            for ( auto freem : item.second ) {
                if (freem != item.second.back()) {
                    outfile << (double)freem/mb << "," << std::flush;
                } else {
                    outfile << (double)freem/mb << std::endl;
                }
            }

        }

        outfile.close();

        outfile.open(mempctfileOut, std::ios::out);
        outfile << "code_location,";
        for (i=0; i<meminfo_names.size(); i++) {
            outfile << i << ",";
        }
        outfile << "Total" << std::endl;
        for (const auto& item_pct : freemem_pct) {
            outfile << item_pct.first << ",";
            // Can't use an iterator here or it'll break the 
            // csv if a numanode percent == totalpercent.
            for (i=0; i<item_pct.second.size(); i++ ) {
                freem_pct = item_pct.second.at(i);
                if (i == (item_pct.second.size()-1)) {
                    outfile << freem_pct << std::endl;
                } else {
                    outfile << freem_pct << ",";
                }
            }

        }

        outfile.close();
    }
    MPI_Barrier(eachnode);
}

void MemoryRecorder::write_rss(int filewrite) {

    // Get the RSS MAX and summarize it if you haven't already done so.
    if (getrss_summary == 0) {
        this->summarizeMaxRSS();
    }

    std::ios_base::fmtflags fls;
    fls = std::cout.flags();
    std::cout << std::fixed << std::dec << std::setprecision(4);
    if (localrank == 0) {
        unsigned long noderam = this->getRamSize()/kb;
        unsigned long totalram = noderam * numnodes;
        double pctnoderam = (double)local_maxrss.val/noderam;

        // Write out Node rss max sum and per rank value.
        if (filewrite) {
            std::ofstream outfile;
            outfile.open(rssfileOut, std::ios::out);
            outfile << std::setprecision(4);

            outfile << "Node Number " << nodenum << "/" << numnodes << "  " << local_maxrss.index << std::endl;
            outfile << "Mem Used: " << local_maxrss.val << " - " << local_maxrss.val/mb << " (GiB)" << std::endl;
            outfile << "Total Ram: " << noderam/mb << " (GiB)" << std::endl;
            outfile << "Fraction Ram Used: " << pctnoderam << std::endl;
            outfile << "Percent Ram Used: " << round_pct(pctnoderam) << "%" << std::endl;

            // Write out Node MaxRSS for each process.
            for (int i=0; i<localsize; i++) {
                outfile << "Rank: " << i << " MaxRSS: " << rss_collect[i]/kb << " (MiB)" << std::endl;
            }
            outfile.close();
        }

        // Write out total program rss max.
        if (globalrank == 0) {
            double pcttotalram = (double)global_maxrss/totalram;
            double pctminram = (double)min_maxrss.val/noderam;
            double pctmaxram = (double)max_maxrss.val/noderam;
            // PRINT TOTAL
            std::cout << "Mem Used: " << global_maxrss << " Total Ram: " << totalram << " Fraction Ram: ";
            std::cout << round_pct(pcttotalram) << "%" << std::endl;
            // PRINT TOTAL SUMMARY
            std::cout << "TOTAL RSS MAX: " << global_maxrss/mb  << " (GiB) - ";
            std::cout << round_pct(pcttotalram) << "%" << std::endl;
            // PRINT MIN
            std::cout << "MIN RSS MAX: " << min_maxrss.val << " " << min_maxrss.val/mb  << " (GiB) - ";
            std::cout << round_pct(pctminram) << "%" << " -- On NODE: " << min_maxrss.index;
            std::cout << " - " << hostlist.at(min_maxrss.index) << std::endl;
            // PRINT MAX
            std::cout << "MAX RSS MAX: " << max_maxrss.val << " " << max_maxrss.val/mb  << " (GiB) - ";
            std::cout << round_pct(pctmaxram) << "%" << " -- On NODE: " << max_maxrss.index;
            std::cout << " - " << hostlist.at(max_maxrss.index) << std::endl;
        }
    }
    std::cout.flags(fls);
}


MemoryRecorder::~MemoryRecorder() {
    free(rss_collect);
}