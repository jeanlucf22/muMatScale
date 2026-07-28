muMatScale OpenMP Offload Patch Effects Summary
===============================================

Scope
-----
This file describes the patch files currently present in the patches/
directory. Patch 05 remains proposed only. Patches 01 through 04 and patches
06 through 17 are applied based on the prior patch application history and the
current applied sequence. Patch 18 was applied and then reverted after it
degraded measured application performance. The duplicate curvature follow-up
patch previously numbered 19 was removed; patch 05 is the canonical
curvature-readiness patch.


Category Index
--------------

Performance improvement:
  02-growth-collapse3-offload.patch
  03-grain-activation-sparse-sync.patch
  04-halo-buffer-map-alloc.patch
  05-curvature-offload-readiness.patch
  07-contiguous-halo-buffer-slabs.patch
  08-batched-halo-pack-unpack-slabs.patch
  09-grouped-halo-completion.patch
  10-batched-grain-cell-device-stamp.patch
  13-frontier-openmp-device-selection.patch
  14-frontier-cartesian-reorder-option.patch
  15-direct-contiguous-halo-device-mpi.patch
  16-active-face-span-cpu-mpi-transfers.patch
  17-progressive-halo-completion-unpack.patch

Reverted / not active performance experiment:
  18-row-prefix-growth-worklists.patch

Correctness / device-host consistency:
  01-checkpoint-device-sync.patch
  03-grain-activation-sparse-sync.patch
  05-curvature-offload-readiness.patch

Correctness / control-flow safety:
  12-nucleation-layer-dc-capture-correctness.patch

Build / portability / configuration cleanup:
  06-cmake-offload-flags.patch
  13-frontier-openmp-device-selection.patch
  14-frontier-cartesian-reorder-option.patch

Refactoring / enabling infrastructure:
  07-contiguous-halo-buffer-slabs.patch
  08-batched-halo-pack-unpack-slabs.patch
  09-grouped-halo-completion.patch
  10-batched-grain-cell-device-stamp.patch
  15-direct-contiguous-halo-device-mpi.patch
  17-progressive-halo-completion-unpack.patch

Transfer reduction / OpenMP mapping cleanup:
  03-grain-activation-sparse-sync.patch
  04-halo-buffer-map-alloc.patch
  07-contiguous-halo-buffer-slabs.patch
  08-batched-halo-pack-unpack-slabs.patch
  10-batched-grain-cell-device-stamp.patch
  15-direct-contiguous-halo-device-mpi.patch
  16-active-face-span-cpu-mpi-transfers.patch

Frontier topology / runtime placement:
  13-frontier-openmp-device-selection.patch
  14-frontier-cartesian-reorder-option.patch

Observability / profiling:
  11-halo-profile-counters.patch


Patch Details
-------------

01-checkpoint-device-sync.patch
Primary category:
  Correctness / device-host consistency

Secondary category:
  Runtime reliability, checkpoint safety

Files changed:
  src/checkpoint.c

Detailed description:
  Adds a GPU_OMP-only helper named syncCheckpointFieldsFromDevice(). Before
  each subblock checkpoint is written, the helper performs OpenMP target
  updates from device to host for checkpoint-relevant fields such as
  temperature, gr, fs, ce, oce, cl, diff_id, mold, d, nuc_threshold, dc, and
  curv when tip curvature is enabled.

  The helper uses sb->totaldim when available and falls back to the global
  subblock dimensions. It also includes profiler.h and records the transfer as
  OFFLOADING_GPU_CPU.

Effect:
  Prevents checkpoint files from being written with stale host-side data when
  the current values live on the device. This is mainly a correctness and
  restart-safety patch. It can add extra GPU-to-CPU transfer cost at checkpoint
  time, but that cost is localized to checkpoint output.

Dependencies:
  No direct dependency on other patches.

Applied status:
  Applied.


02-growth-collapse3-offload.patch
Primary category:
  Performance improvement

Secondary category:
  OpenMP offload cleanup

Files changed:
  src/growth.c

Detailed description:
  Rewrites several GPU_OMP loops from a split structure using
  "target teams distribute" on the outer k loop plus a nested "parallel for
  collapse(2)" into a single combined OpenMP offload construct:

    #pragma omp target teams distribute parallel for collapse(3) schedule(static,1)

  The affected routines include:
    sb_diffuse_alloy_decentered()
    fs_change_diffuse()
    grow_octahedra()
    grow_cell_reduction()

  grow_cell_reduction() keeps the gindex and nindex mapping but folds it into
  the combined target teams distribute parallel for construct.

Effect:
  Gives the compiler/runtime a single collapsed 3D iteration space for GPU
  execution. This should reduce nested parallel-region overhead, expose more
  work to the device scheduler, and improve occupancy for the 3D stencil-like
  loops.

Dependencies:
  No direct dependency on other patches.

Applied status:
  Applied.


03-grain-activation-sparse-sync.patch
Primary category:
  Performance improvement

Secondary category:
  Transfer reduction, device-host consistency

Files changed:
  src/grain.c

Detailed description:
  Reduces broad device-host transfers in grain creation and activation.

  In createGrains(), the patch removes the full gr field update from device to
  host before grain creation and removes the full gr field update back to the
  device after grain creation. Instead, each newly created grain cell initially
  receives a targeted device update for gr[idx:1].

  In activateNewGrains(), the patch changes the grain_cache update from the
  whole allocated grain cache to only the newly activated slice:

    grain_cache[bp->num_grains:new_activations]

  In cell_nucleation(), the patch updates only the valid newGrainLocs entries,
  and only when numNewGrains is greater than zero.

Effect:
  Cuts large CPU-GPU transfers down to sparse updates proportional to the
  number of newly created or activated grains. This is especially helpful when
  the grain count changes by a small amount relative to total grid or cache
  size.

Dependencies:
  Patch 10 builds on this area and later replaces the per-grain gr[idx:1]
  update with a batched device-side stamping kernel.

Applied status:
  Applied, but partly superseded by patch 10.


04-halo-buffer-map-alloc.patch
Primary category:
  Performance improvement

Secondary category:
  OpenMP mapping cleanup, transfer reduction

Files changed:
  src/calculate.c

Detailed description:
  Changes halo receive/send buffer OpenMP mappings from map(to:...) to
  map(alloc:...) under GPU_PACK. These buffers are scratch buffers, so copying
  the host memset-initialized bytes to the device is unnecessary.

  The patch covers the supported halo data sizes:
    8-byte double fields
    4-byte int fields
    24-byte three-double fields such as decentered vectors

Effect:
  Avoids needless host-to-device transfer during halo buffer setup. This lowers
  startup/setup transfer volume and makes the intended scratch-buffer ownership
  clearer to the OpenMP runtime.

Dependencies:
  Patch 07 builds on and supersedes this by replacing per-face buffer mappings
  with contiguous send/receive buffer slab mappings.

Applied status:
  Applied, but superseded by patch 07.


05-curvature-offload-readiness.patch
Primary category:
  Performance improvement

Secondary category:
  Correctness / device data readiness

Files changed:
  src/setup.c
  src/calculate.c
  src/curvature.c

Detailed description:
  Prepares the curvature calculation for OpenMP offload.

  In setup.c, the patch maps lsp->curv to the device when bp->tip_curv is
  enabled. In calculate.c, it adds an sb_curvature() call in the iteration flow
  when tip curvature is active. In curvature.c, it adds a GPU_OMP target teams
  distribute parallel for collapse(3) construct around the main curvature loop,
  including a large private variable list for the loop temporaries.

Effect:
  Moves curvature work toward device execution and ensures the curv array is
  present on the device before it is produced/consumed by offloaded code. The
  likely performance benefit comes from avoiding CPU-side curvature work in a
  GPU build, but the exact benefit depends on whether curvature is enabled and
  how frequently it is computed.

Dependencies:
  No direct dependency on other listed patches.

Applied status:
  Not applied/proposed only.


06-cmake-offload-flags.patch
Primary category:
  Build / portability / configuration cleanup

Secondary category:
  Build correctness

Files changed:
  CMakeLists.txt
  tests/CMakeLists.txt

Detailed description:
  Changes how EXTRA_OpenMP_C_FLAGS are passed to the build. Instead of appending
  the extra flags globally to CMAKE_C_FLAGS and passing them through
  target_link_libraries(), the patch parses them with separate_arguments() and
  applies them with:

    target_compile_options(...)
    target_link_options(...)

  The same approach is applied to the testPacking target.

Effect:
  Keeps compiler/linker flags out of the library list and makes GPU offload
  flags target-scoped. This improves portability across CMake generators and
  avoids confusing link flags with actual libraries.

Dependencies:
  No source-level dependency on other patches.

Applied status:
  Applied.


07-contiguous-halo-buffer-slabs.patch
Primary category:
  Performance improvement

Secondary category:
  Refactoring / enabling infrastructure

Files changed:
  src/calculate.c

Detailed description:
  Refactors halo communication buffer allocation. The variable_registration
  structure gains:

    rbuf_base
    sbuf_base
    buffer_slot_cells
    buffer_slot_bytes

  Instead of allocating and mapping six independent receive buffers and six
  independent send buffers per variable, the patch allocates one contiguous
  receive slab and one contiguous send slab. The per-face rbuf[face] and
  sbuf[face] pointers become offsets into those slabs.

  Under GPU_PACK, the patch maps each whole slab once with map(alloc:...).

Effect:
  Reduces allocation and OpenMP mapping overhead, improves buffer locality, and
  creates the memory layout needed for later batched face pack/unpack kernels.

Dependencies:
  Builds on the halo buffer mapping area changed by patch 04. Patch 08 depends
  on this contiguous slab layout.

Applied status:
  Applied.


08-batched-halo-pack-unpack-slabs.patch
Primary category:
  Performance improvement

Secondary category:
  Refactoring / GPU launch batching

Files changed:
  src/calculate.c
  src/face_util.c
  src/face_util.h
  src/packing.c
  src/packing.h

Detailed description:
  Adds batched halo packing and unpacking for the contiguous buffer slabs from
  patch 07.

  The patch introduces pack_faces_field() and unpack_faces_field(), plus
  type-specialized implementations for:

    double fields
    int fields
    three-double fields

  SendFacesNB() now first identifies active remote faces, computes face packing
  metadata for each one, and calls pack_faces_field() once for the active face
  set. Send_Plane() no longer performs per-face packing.

  FinishExchangeForVar() similarly collects active halo faces and calls
  unpack_faces_field() once for the active set.

  SendRecvHalosNB() receives the buffer_slot_cells argument so the batched
  packing code can index into the contiguous per-face slots inside the slab.

Effect:
  Reduces GPU kernel launch count by packing or unpacking multiple faces in one
  launch per variable instead of launching once per face. This directly targets
  the halo exchange bottleneck where small per-face kernels can spend too much
  time in launch overhead relative to useful work.

Dependencies:
  Depends on patch 07. Patch 09 and patch 11 both build on this API and control
  flow.

Applied status:
  Applied. Later patches modify the same regions, so exact reverse-apply checks
  may conflict even though the effect is present.


09-grouped-halo-completion.patch
Primary category:
  Performance improvement

Secondary category:
  Refactoring / communication-computation scheduling

Files changed:
  src/calculate.c

Detailed description:
  Splits the previous FinishExchangeForVar() behavior into separate wait and
  unpack stages:

    WaitExchangeForVar()
    UnpackExchangeForVar()
    FinishExchangeForVar()
    FinishExchangeForVars()

  FinishExchangeForVars() waits for all listed variables first, then performs
  all corresponding unpack operations. The iteration flow uses this grouped
  completion for cl/fs and dc/d.

Effect:
  Avoids serializing each variable as "wait, unpack, wait, unpack". Grouping
  waits before unpacking should improve scheduling around MPI completion and
  device unpack work, especially after patch 08 made unpacking a larger batched
  operation.

Dependencies:
  Depends on patch 08 because it refactors the batched unpack path introduced
  there.

Applied status:
  Applied.


10-batched-grain-cell-device-stamp.patch
Primary category:
  Performance improvement

Secondary category:
  Transfer reduction, GPU launch batching, refactoring

Files changed:
  src/grain.c

Detailed description:
  Adds syncNewGrainCellsToDevice(), which batches device-side updates of the gr
  field for newly created grains.

  The function first updates the compact newGrainLocs array to the device, then
  launches a single OpenMP target teams distribute parallel for over the new
  grain count. Each device thread computes the grid index for one new grain
  location and stamps the appropriate grain id into gr.

  This replaces the earlier per-grain:

    #pragma omp target update to(gr[idx:1])

  calls inside createGrains().

Effect:
  Converts many tiny CPU-to-GPU updates into one compact location transfer plus
  one device kernel. This should reduce transfer and launch overhead when many
  grains are created in a timestep.

Dependencies:
  Depends on the sparse grain update path from patch 03.

Applied status:
  Applied.


11-halo-profile-counters.patch
Primary category:
  Observability / profiling

Secondary category:
  Performance analysis support

Files changed:
  src/face_util.c
  src/packing.c
  src/profiler.c
  src/profiler.h

Detailed description:
  Adds halo-specific counters to the profiler. The new counters track:

    Pack launches
    Unpack launches
    Pack bytes
    Unpack bytes
    CPU-MPI D2H bytes
    CPU-MPI H2D bytes
    Active send faces
    Active recv faces

  The patch adds the halo_counter_tag enum, profiler_count_halo(), per-rank
  counter storage, MPI gathering for halo counters, console reporting, and a
  CSV output file named:

    <basefilename>_halo_profile_<timestep>.csv

  It instruments both the original single-face packing/unpacking helpers and
  the batched face helpers from patch 08.

Effect:
  Does not directly optimize runtime behavior. Instead, it provides measurement
  data needed to verify whether the halo batching patches reduce launch counts
  and transfer volume as expected.

Dependencies:
  Depends on patch 08 for the batched halo helper instrumentation. It can still
  count legacy helper paths, but its most important validation value is tied to
  the batched halo path.

Applied status:
  Applied.


12-nucleation-layer-dc-capture-correctness.patch
Primary category:
  Correctness / control-flow safety

Secondary category:
  Defensive cleanup, GPU correctness

Files changed:
  src/grain.c
  src/growth.c

Detailed description:
  Fixes two correctness hazards found during the Frontier-oriented review.

  In src/grain.c, the patch corrects a misspelled variable in the non-NUC_SEP
  nucleation path:

    lyaerno -> layerno

  The current default build defines NUC_SEP, so this typo is latent in the
  default configuration. It can still break alternate builds where the
  non-NUC_SEP path is compiled.

  In src/growth.c, the patch guards the decentered-octahedron temporary copyback
  in capture_octahedra_diffuse(). The first INDEX_SEP capture loop writes
  dc_tmp[i] only when a candidate cell is actually captured. The later copyback
  loop previously copied dc_tmp[i] into dc[idx] for every candidate, including
  cells that were skipped or not captured. The patch adds:

    if (gr[idx] == ogr[idx])
        continue;

  This means only cells whose grain id changed during capture consume dc_tmp.

Effect:
  Removes a latent compile-time error in an alternate nucleation configuration
  and prevents uninitialized temporary decentered-octahedron values from
  overwriting dc for cells that were not captured. Runtime performance should be
  neutral or slightly positive because the copyback loop skips unnecessary
  writes.

Dependencies:
  No direct dependency on other patches. It applies to the current patched
  source and is most relevant when INDEX_SEP is enabled.

Applied status:
  Applied.


13-frontier-openmp-device-selection.patch
Primary category:
  Frontier topology / runtime placement

Secondary category:
  Performance portability, GPU runtime configuration

Files changed:
  src/ca_main.c

Detailed description:
  Adds a GPU_OMP-only runtime helper that selects the OpenMP target device for
  each MPI rank before any device data mappings are created.

  The helper first honors an explicit MUMATSCALE_OMP_DEVICE setting. If that
  override is absent and OMP_DEFAULT_DEVICE is not already set, it derives a
  device id from scheduler/MPI local-rank environment variables such as:

    SLURM_LOCALID
    OMPI_COMM_WORLD_LOCAL_RANK
    MV2_COMM_WORLD_LOCAL_RANK
    MPI_LOCALRANKID
    PMI_LOCAL_RANK

  The selected device is applied with omp_set_default_device(). Explicit device
  selections are range-checked against omp_get_num_devices().

Effect:
  Makes MPI-rank-to-GPU placement more robust on Frontier-like nodes where a
  single node exposes multiple AMD MI250X GCDs. This reduces the risk that
  multiple ranks accidentally target the same visible OpenMP device when the
  launch environment does not fully mask devices per rank.

Dependencies:
  No source-level dependency on other patches. It should be applied before
  performance runs that rely on multiple ranks per node.

Applied status:
  Applied.


14-frontier-cartesian-reorder-option.patch
Primary category:
  Frontier topology / runtime placement

Secondary category:
  MPI placement configurability

Files changed:
  src/ca_main.c

Detailed description:
  Adds an environment-controlled MPI Cartesian communicator reorder option. The
  existing MPI_Cart_create() call keeps reorder disabled by default. When
  MUMATSCALE_MPI_CART_REORDER is set to a nonzero value, the patch passes
  reorder=1 to MPI_Cart_create().

Effect:
  Allows the MPI implementation and launcher to remap Cartesian ranks when that
  improves locality between neighboring subblocks. On Frontier, this gives Cray
  MPI a chance to align the logical 3D subblock topology with the node and
  network placement chosen for the job, while preserving the current behavior
  unless explicitly enabled.

Dependencies:
  No direct dependency on other patches. It complements patch 13 because both
  address placement, but either patch can be applied independently.

Applied status:
  Applied.


15-direct-contiguous-halo-device-mpi.patch
Primary category:
  Performance improvement

Secondary category:
  GPU-aware MPI communication cleanup

Files changed:
  src/calculate.c
  src/face_util.c
  src/face_util.h
  src/packing.c
  src/packing.h

Detailed description:
  Adds a direct GPU-aware MPI path for contiguous top and bottom halo planes.
  These planes are contiguous in the field layout, so they do not require a
  pack or unpack staging kernel when GPU_PACK is enabled and CPU_MPI is not
  defined.

  The patch adds:

    face_is_contiguous_plane()
    field_plane_ptr()
    use_direct_device_plane()

  Send_Plane() and Recv_Plane() use field_plane_ptr() inside an OpenMP
  use_device_ptr region for direct MPI_Isend()/MPI_Irecv() calls. SendFacesNB()
  skips staged packing for direct planes, and UnpackExchangeForVar() skips
  unpacking for direct planes because the receive already lands in the halo
  field.

Effect:
  Removes avoidable pack/unpack work and staging-buffer traffic for contiguous
  halo planes under GPU-aware MPI. This should reduce small kernel launches and
  memory motion in the halo path, especially for decompositions where top and
  bottom neighbors are remote ranks.

Dependencies:
  Depends on the contiguous halo slab and batched halo helper infrastructure
  from patches 07 and 08. It also assumes the GPU-aware MPI build path, namely
  GPU_PACK without CPU_MPI.

Applied status:
  Applied.


16-active-face-span-cpu-mpi-transfers.patch
Primary category:
  Performance improvement

Secondary category:
  Transfer reduction / CPU-MPI fallback cleanup

Files changed:
  src/packing.c

Detailed description:
  Adds packed_faces_buffer_span(), which computes the smallest contiguous range
  of face slots containing the active faces in a batched halo operation.

  The CPU_MPI fallback path previously updated the full six-face staging slab
  after packing or before unpacking:

    buffer[0:NUM_NEIGHBORS * buffer_slot_cells]

  This patch changes those OpenMP target updates to cover only:

    buffer[update_start:update_len]

  where update_start and update_len are derived from the active face ids.

Effect:
  Reduces device-host and host-device copy volume in CPU-MPI fallback builds
  when fewer than all six face slots need staging. It also combines naturally
  with patch 15: if direct top/bottom planes are removed from staging, the
  remaining side-face slots are copied as a tighter span.

Dependencies:
  Depends on the batched halo staging layout introduced by patches 07 and 08.
  The current patch text also assumes the halo profiler counters from patch 11
  are present. It can be applied before or after patch 15.

Applied status:
  Applied.


17-progressive-halo-completion-unpack.patch
Primary category:
  Performance improvement

Secondary category:
  Communication-computation scheduling

Files changed:
  src/calculate.c

Detailed description:
  Adds TestExchangeForVar(), a nonblocking MPI_Testall() wrapper for one halo
  variable. FinishExchangeForVars() is changed from a strict two-phase sequence:

    wait all listed variables, then unpack all listed variables

  into a progressive completion loop. Each variable is unpacked as soon as its
  own outstanding MPI requests complete. If polling finds no newly completed
  variable, the code falls back to WaitExchangeForVar() for the next unfinished
  variable.

Effect:
  Avoids forcing unpack work to wait for the slowest variable in a group when
  another variable has already completed. On Frontier-scale runs this may expose
  a little more overlap between MPI progress and device unpack kernels,
  especially when halo sizes or network paths differ across fields.

Dependencies:
  Depends on the grouped halo completion structure introduced by patch 09.
  It complements patches 15 and 16 but does not require them.

Applied status:
  Applied.


18-row-prefix-growth-worklists.patch
Primary category:
  Reverted / not active performance experiment

Secondary category:
  Atomic reduction / GPU worklist refactoring

Files changed:
  src/growth.c

Detailed description:
  Replaces the GPU_OMP path in grow_cell_reduction() with a row-count,
  prefix-sum, and fill sequence.

  The original GPU path used global atomic capture operations to append
  candidate cells to diff_id and, when NUC_PRELIST is enabled, nuc_id. This
  patch counts candidates per (z,y) row on the GPU, copies the compact row
  counts back to the CPU, computes row offsets, copies the offsets to the GPU,
  and then fills the worklists without device-wide atomic increments. The CPU
  path is preserved.

Effect:
  Intended to reduce global atomic serialization in the growth/capture
  worklist builder. On the measured application run, this degraded performance,
  most likely because the added count/fill kernels and count/offset transfers
  introduced synchronization and launch overhead that outweighed the reduction
  in atomic contention. The active source has therefore been restored to the
  prior single-pass GPU_OMP append path.

Dependencies:
  Depends on patch 02 because it replaces the current GPU_OMP
  grow_cell_reduction() loop structure. It is independent of patch 12.

Applied status:
  Reverted / not active.


Applied Patch Dependency Summary
--------------------------------

Independent applied patches:
  01
  06
  12
  13
  14

Applied dependency chains:
  03 -> 10
  04 -> 07 -> 08 -> 09 -> 17
  04 -> 07 -> 08 -> 11 -> 16
  04 -> 07 -> 08 -> 15

Applied dependency notes:
  Patch 15 requires the batched/slab halo infrastructure from patches 07 and
  08, and it is only active in GPU-aware MPI builds.

  Patch 16 is written against the profiled batched halo code and assumes patch
  11 counters are present. It can be applied before or after patch 15.

  Patch 17 depends on the grouped completion structure from patch 09. It does
  not require patches 15 or 16, but they are complementary.

Remaining proposed patches:
  05

Reverted / not active patches:
  18

Notes for remaining proposed patches:
  Patch 05 is not applied. It has no direct source-level dependency on the
  other listed patches. Patch 01 remains useful when checkpointing curv if
  patch 05 is applied.

Notes for reverted / not active patches:
  Patch 18 depends on patch 02 because it replaces the GPU_OMP
  grow_cell_reduction() loop shape introduced there. It has been reverted from
  the active source after degrading measured application performance.
