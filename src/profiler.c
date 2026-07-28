/***************************************************************/
/* Copyright (c) 2023, Lang Yuan, Univeristy of South Carolina */
/* All rights reserved.                                        */
/* This file is part of muMatScale.                            */
/* See the top-level LICENSE file for details.                 */
/***************************************************************/

#include <mpi.h>
#include "debug.h"
#include "xmalloc.h"
#include "profiler.h"
#include "globals.h"
#include "functions.h"

static char *bucket_names[] = {
    "Setup",
    "Temp Prep",
    "Chkpt",
    "HDF Output",
    "Output",
    "Face Xchg (local)",
    "Face Xchg SEND",
    "Face Xchg WAIT",
    "Temp Update",
    "Nucleation",
    "FS Change",
    "Grow Octahedra",
    "Cell Index",
    "Capture Octahedra",
    "Diffuse Alloy",
    "Grain Activation",
    "Grain Sync 1",
    "Grain Sync 2",
    "Offloading CPU-GPU",
    "Offloading GPU-CPU",
    "Offloading IO",
    "Packing CPU-GPU",
    "Packing GPU-CPU",
    "Reduce fs",
    "unPacking",
    "Packing",
    "Initialization",
    "Computation",
    "Communication",
    "Local Exchange",
    "Synchronization",
    "File IO",
};

static char *halo_counter_names[] = {
    "Pack launches",
    "Unpack launches",
    "Pack bytes",
    "Unpack bytes",
    "CPU-MPI D2H bytes",
    "CPU-MPI H2D bytes",
    "Active send faces",
    "Active recv faces",
};

static double *bucket_times = NULL;
static unsigned long long halo_counters[NUM_HALO_COUNTERS] = { 0 };
static double init_time = 0.0;
static double last_recorded = 0.0;

void
profiler_init(
    )
{
    /* Verify that our sizes are the same */
    assert(sizeof(bucket_names) / sizeof(char *) == NUM_BUCKETS);
    assert(sizeof(halo_counter_names) / sizeof(char *) == NUM_HALO_COUNTERS);

    /* Each process stores timing info into the bucket array entries */
    xmalloc(bucket_times, double,
            NUM_BUCKETS);
    for (int i = 0; i < NUM_HALO_COUNTERS; i++)
    {
        halo_counters[i] = 0;
    }

    init_time = MPI_Wtime();
    last_recorded = init_time;
}

void
profiler_count_halo(
    halo_counter_tag counter,
    uint64_t amount)
{
    if (counter >= 0 && counter < NUM_HALO_COUNTERS)
    {
        halo_counters[counter] += (unsigned long long) amount;
    }
}


void
profile(
    bucket_tag tag)
{
    double time_now = MPI_Wtime();
    bucket_times[tag] += (time_now - last_recorded);
    last_recorded = time_now;
}

/* 	[__PROFILE]
	Accumulate the given elapsed time in one of the bucket
	- Initialization, Computation, Communication, and fileIO
*/
void
timing(
    bucket_tag tag,
    double t)
{
    timer_stop();
    bucket_times[tag] += t;
    timer_start();
}

void
profiler_collate(
    )
{
#ifndef PATH_MAX
#define PATH_MAX 512
#endif
    double (
    *allbuckets)[NUM_BUCKETS] = NULL;
    unsigned long long (
    *all_halo_counters)[NUM_HALO_COUNTERS] = NULL;

    if (iproc == 0)
    {
        allbuckets =
            (double (*)[NUM_BUCKETS]) malloc(sizeof(double[NUM_BUCKETS]) *
                                             nproc);
        all_halo_counters =
            (unsigned long long (*)[NUM_HALO_COUNTERS])
            malloc(sizeof(unsigned long long[NUM_HALO_COUNTERS]) * nproc);
    }
    MPI_Gather(bucket_times, NUM_BUCKETS, MPI_DOUBLE,
               allbuckets, NUM_BUCKETS, MPI_DOUBLE, 0, mpi_comm_new);
    MPI_Gather(halo_counters, NUM_HALO_COUNTERS, MPI_UNSIGNED_LONG_LONG,
               all_halo_counters, NUM_HALO_COUNTERS, MPI_UNSIGNED_LONG_LONG,
               0, mpi_comm_new);

    if (iproc == 0)
    {
        char profiler_file[PATH_MAX] = { 0 };
        snprintf(profiler_file, PATH_MAX - 1, "%s_profile_%lu.csv",
                 bp->basefilename, bp->timestep);
        FILE *fp = fopen(profiler_file, "w");

        int j = 0;
        fprintf(fp, "Rank,Num SB,");
        for (; j < (NUM_BUCKETS - SYS_PROFILE); j++)
        {
            fprintf(fp, "%s,", bucket_names[j]);
        }
        fprintf(fp, "Total,");

        for (; j < NUM_BUCKETS; j++)
        {                       //[__PROFILE]
            fprintf(fp, "%s,", bucket_names[j]);
        }
        fprintf(fp, "System Total\n");


        for (int i = 0; i < nproc; i++)
        {
            j = 0;
            fprintf(fp, "%d,%d,", i, 1);
            double ranksum = 0.0;
            for (; j < (NUM_BUCKETS - SYS_PROFILE); j++)
            {
                ranksum += allbuckets[i][j];
                fprintf(fp, "%lg,", allbuckets[i][j]);
            }
            fprintf(fp, "%lg,", ranksum);
            ranksum = 0.0;
            for (; j < NUM_BUCKETS; j++)
            {                   //[__PROFILE]
                ranksum += allbuckets[i][j];
                fprintf(fp, "%lg,", allbuckets[i][j]);
            }
            fprintf(fp, "%lg\n", ranksum);
        }
        fclose(fp);

        char halo_file[PATH_MAX] = { 0 };
        snprintf(halo_file, PATH_MAX - 1, "%s_halo_profile_%lu.csv",
                 bp->basefilename, bp->timestep);
        fp = fopen(halo_file, "w");
        fprintf(fp, "Rank");
        for (int j = 0; j < NUM_HALO_COUNTERS; j++)
        {
            fprintf(fp, ",%s", halo_counter_names[j]);
        }
        fprintf(fp, "\n");
        for (int i = 0; i < nproc; i++)
        {
            fprintf(fp, "%d", i);
            for (int j = 0; j < NUM_HALO_COUNTERS; j++)
            {
                fprintf(fp, ",%llu", all_halo_counters[i][j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);

        xfree(allbuckets);
        xfree(all_halo_counters);
    }
}


void
profiler_write_stats(
    )
{
    double accum_time = 0.0;
    double time_now = MPI_Wtime();
    printf("Timing Statistics for rank %d\n", iproc);
    for (int i = 0; i < NUM_BUCKETS; i++)
    {
        accum_time += bucket_times[i];
        printf("%40s     %lg\n", bucket_names[i], bucket_times[i]);
    }
    for (int i = 0; i < NUM_HALO_COUNTERS; i++)
    {
        printf("%40s     %llu\n", halo_counter_names[i], halo_counters[i]);
    }
    double total_time = time_now - init_time;
    printf("Total time:  %lg\n", total_time);
    printf("Accumulated time:  %lg\n", accum_time);
    printf("Unaccounted for time:  %lg\n", total_time - accum_time);
}
