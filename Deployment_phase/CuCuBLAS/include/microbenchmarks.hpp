///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header for the CoCoPeLia Microbenchmarks (Deployment_phase).
///

#ifndef MICROBENCH_H
#define MICROBENCH_H

/// Check for BOOST use (enabled from CmakeFiles) or number of ITER for microbenchmarks.
#ifdef AUTO_BENCH_USE_BOOST
#include <boost/math/distributions/students_t.hpp>
#define alphaCI 0.05
#define MICRO_MIN_ITER 10
#define MICRO_MAX_ITER 100000
#else
#ifndef ITER
#error
#endif
#endif

/// The minimum dimensions and step sizes (of Dim*Dim chunks) for microbenchmarks
#define MIN_DIM_TRANS 256
#define STEP_TRANS 256
#define MAX_DIM_TRANS 10000

#define MIN_DIM_BLAS3 256
#define STEP_BLAS3 256
#define MAX_DIM_BLAS3 10000

#define MIN_DIM_BLAS1 4096
#define STEP_BLAS1 4096

#include <stdio.h>
#include <cstring>
#include <cuda.h>
#include "cublas_v2.h"

#include "unihelpers.hpp"

#endif
