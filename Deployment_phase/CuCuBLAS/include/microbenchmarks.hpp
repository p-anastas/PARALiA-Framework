///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The headers for Micro-benchmark limits, step, etc
///

#ifndef MICROBENCH_H
#define MICROBENCH_H

#ifdef AUTO_BENCH_USE_BOOST
#include <boost/math/distributions/students_t.hpp>
#else
#ifndef ITER
#error
#endif
#endif

#define alphaCI 0.05
#define MICRO_MIN_ITER 10
#define MICRO_MAX_ITER 100000

#define MIN_DIM_TRANS 256
#define STEP_TRANS 256

#define MIN_DIM_BLAS3 256
#define STEP_BLAS3 256

#define maxDim_blas2 16384
#define minDim_blas2 256
#define step_blas2 256

#define MIN_DIM_BLAS1 256
#define STEP_BLAS1 256

#define maxDim_blas1 268435456
#define minDim_blas1 256
#define N_step_blas1 256

#include <stdio.h>
#include <cstring>
#include <cuda.h>
#include "cublas_v2.h"

#include "unihelpers.hpp"

#endif
