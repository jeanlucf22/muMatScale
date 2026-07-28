/***************************************************************/
/* Copyright (c) 2023, Lang Yuan, Univeristy of South Carolina */
/* All rights reserved.                                        */
/* This file is part of muMatScale.                            */
/* See the top-level LICENSE file for details.                 */
/***************************************************************/

#ifndef PROFILER_H__
#define PROFILER_H__

#include <stdint.h>

#include "stopwatch.h"
#define SYS_PROFILE 6           //number of system profile metrics, see [1-6]

typedef enum
{
    SETUP = 0,
    TEMPERATURE_PREPARE,
    PROF_CHECKPOINT,
    HDF_OUTPUT,
    PROF_OUTPUT,
    FACE_EXCHNG_LOCAL,
    FACE_EXCHNG_REMOTE_SEND,
    FACE_EXCHNG_REMOTE_WAIT,
    TEMP_UPDATE,
    CALC_NUCLEATION,
    CALC_FS_CHANGE,
    CALC_GROW_OCTAHEDRA,
    CALC_CELL_INDEX,
    CALC_CAPTURE_OCTAHEDRA,
    CALC_DIFFUSE_ALLOY,
    GRAIN_ACTIVATION,
    GRAIN_ACTIV_SYNC1,
    GRAIN_ACTIV_SYNC2,
    OFFLOADING_CPU_GPU,
    OFFLOADING_GPU_CPU,
    OFFLOADING_IO,
    PACKING_CPU_GPU,
    PACKING_GPU_CPU,
    REDUCE_FS,
    UNPACKING,
    PACKING,
    INITIALIZATION,             //[1] Application initialization time
    COMPUTATION,                //[2] Total computation time (CPU)
    COMMUNICATION,              //[3] Total remote communication time (MPI remote)
    LOCAL_EX,                   //[4] Time on local data exchange (MPI_Sendrecv)
    SYNC_TS,                    //[5] Synchronization between all processes (MPI_Gather)
    FILEIO,                     //[6] Total file I/O time (DISK)
    NUM_BUCKETS,                /* NOT AN ACTUAL BUCKET */
    CALC_CURVATURE              //ADD_CURV_LY
} bucket_tag;

typedef enum
{
    HALO_PACK_LAUNCHES = 0,
    HALO_UNPACK_LAUNCHES,
    HALO_PACK_BYTES,
    HALO_UNPACK_BYTES,
    HALO_CPU_MPI_D2H_BYTES,
    HALO_CPU_MPI_H2D_BYTES,
    HALO_ACTIVE_SEND_FACES,
    HALO_ACTIVE_RECV_FACES,
    NUM_HALO_COUNTERS
} halo_counter_tag;

void profiler_init(
    void);
void profile(
    bucket_tag bucket);
void timing(
    bucket_tag tag,
    double t);
void profiler_collate(
    void);
void profiler_write_stats(
    void);
void profiler_count_halo(
    halo_counter_tag counter,
    uint64_t amount);

#endif
