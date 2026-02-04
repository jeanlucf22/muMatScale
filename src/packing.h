/***************************************************************/
/* Copyright (c) 2023, Lang Yuan, Univeristy of South Carolina */
/* All rights reserved.                                        */
/* This file is part of muMatScale.                            */
/* See the top-level LICENSE file for details.                 */
/***************************************************************/

void
unpack_3double(
    double *data,
    const int stride,
    const int bsize,
    const int nblocks,
    const int offset,
    double *buffer);

void
unpack_double(
    double *data,
    const int stride,
    const int bsize,
    const int nblocks,
    const int offset,
    double *buffer);

void
unpack_int(
    int *data,
    const int stride,
    const int bsize,
    const int nblocks,
    const int offset,
    int *buffer);

void pack_field(
    const size_t datasize,
    void *data,
    const int stride,
    const int bsize,
    const int nblocks,
    const int offset,
    void *buffer);

void unpack_field(
    const size_t datasize,
    void *data,
    const int stride,
    const int bsize,
    const int nblocks,
    const int offset,
    void *buffer);

void computeHaloInfo(
    const int halo,
    int *offset,
    int *stride,
    int *bsize,
    int *nblocks);

void computeFaceInfo(
    const int face,
    int *offset,
    int *stride,
    int *bsize,
    int *nblocks);
