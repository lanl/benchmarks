#include <stdio.h>
#include <stdlib.h>


#define NUM_NUMADOMAINS 8
#define NUM_CPUS_PER_NUMADOMAIN 28
#define OFFSET_NUMADOMAIN 14

typedef unsigned int ProcQuant;

static const ProcQuant cpu_preferred_order[NUM_CPUS_PER_NUMADOMAIN] = {13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 112, 
    125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 0};
static const ProcQuant numadomain_preferred_order[NUM_NUMADOMAINS] = {3, 7, 2, 6, 1, 5, 0, 4};
static const ProcQuant numa_domain_start[NUM_NUMADOMAINS] = {0, 14, 28, 42, 56, 70, 84, 98};
static ProcQuant ranks_per_numa_domain[NUM_NUMADOMAINS] = {0, 0, 0, 0, 0, 0, 0, 0};


void load_balance(ProcQuant ranks_per_node)
{
    ProcQuant each = ranks_per_node / NUM_NUMADOMAINS;
    ProcQuant remainder = ranks_per_node % NUM_NUMADOMAINS;
    ProcQuant i;

    for (i = 0; i < NUM_NUMADOMAINS; ++i)
    {
        ranks_per_numa_domain[i] = each;
    }
    for (i = 0; i < remainder; ++i)
    {
        ++ranks_per_numa_domain[numadomain_preferred_order[i]];
    }
    fprintf(stderr, "ranks_per_numa_domain[%hu] = {", NUM_NUMADOMAINS);
    for (i = 0; i < NUM_NUMADOMAINS; ++i)
    {
        if (i != 0)
        {
            fprintf(stderr, ", ");
        }
        fprintf(stderr, "%hu", ranks_per_numa_domain[i]);
    }
    fprintf(stderr, "}\n");
}

int cmp_func(const void *a, const void *b)
{
    return ( *(int*)a - *(int*)b );
}

void cpu_list(ProcQuant *cpus)
{
    ProcQuant i, j, k = 0;
    ProcQuant *cpus_domain;
    for (i = 0; i < NUM_NUMADOMAINS; ++i)
    {
        cpus_domain = (ProcQuant *)malloc(ranks_per_numa_domain[i] * sizeof(ProcQuant));
        for (j = 0; j < ranks_per_numa_domain[i]; ++j)
        {
            cpus_domain[j] = cpu_preferred_order[j] + i * OFFSET_NUMADOMAIN;
        }
        qsort(cpus_domain, ranks_per_numa_domain[i], sizeof(ProcQuant), cmp_func);
        for (j = 0; j < ranks_per_numa_domain[i]; ++j)
        {
            cpus[k] = cpus_domain[j];
            ++k;
        }
        free(cpus_domain);
    }
}

int main (int argc, char **argv)
{
    ProcQuant ranks_per_node = 0;
    char *endptr;  /* strtol */
    ProcQuant *cpus;
    ProcQuant i;

    if (argc != 2)
    {
        fprintf(stderr, "Incorrect number of command line arguments!\n");
        return 1;
    }

    ranks_per_node = (ProcQuant)strtol(argv[1], &endptr, 10);
    fprintf(stderr, "ranks_per_node = %hu\n", ranks_per_node);

    cpus = (ProcQuant *)malloc(ranks_per_node * sizeof(ProcQuant));

    load_balance(ranks_per_node);
    cpu_list(cpus);

    fprintf(stderr, "CPUs:\n");
    for (i = 0; i < ranks_per_node ; ++i)
    {
        if (i > 0)
        {
            fprintf(stdout, ",");
        }
        fprintf(stdout, "%hu", cpus[i]);
    }

    free(cpus);
    return 0;
}
