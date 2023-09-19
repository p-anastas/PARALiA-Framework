# PARALiA-Framework

PARALiA is an end-to-end BLAS framework offering near-optimal BLAS performance and resource utilization using a a model-assisted approach.
It performs microbenchmarks during installation, uses these for constructing BLAS and interconnect performance models, and employs them for autotuning coupled with an optimized task scheduler, resulting in near-optimal data distribution and performance-aware resource utilization.

![PARALiA_Framework_new-new-model](https://user-images.githubusercontent.com/47385258/223406202-a19f5c2e-232c-435a-821f-cbdd77c364f2.jpg)

## Software/Compiler requirements
 - Python 3.x (packages: argsparse, os, fnmatch, pandas, sklearn. Additionally for plotting: math, numpy, scipy, matplotlib, seaborn)
 - CUDA toolkit 10+ (tested with multiple 11.x versions on Tesla V100, A100 and GTX GPUs)
 - A gcc/g++ compiler compatible with the above CUDA (tested with 8, 11).
 - OpenBLAS, installed (preferably) with the same gcc compiler.

## Installation
PARALiA installation consists of 5 easy steps:
 - Duplicate *config.sh* and fill *config_sysname.sh* with deployed system details for each different system.
 - source *config_sysname.sh*
 - (Optional - for some systems) Modify *Deploy.in* with any module loads/link paths etc required for building the DB (gcc, CUDA/CUBLAS, OpenBLAS, python)  
 - mkdir *sysname-build* && cd *sysname-build*
 - cmake ../ && make -j
 - Run *Deploy.sh* in the installation dir (default: *sysname-build/sysname-install*) to perform microbenchmarks and build the database (will take a while to complete).

## Usage
After a succesful installation, PARALiA should provide:
  - **paralia.so**, a shared library in *sysname-install/Library_scheduler/lib* containing all PARALiA BLAS functions.
  - **PARALiA.hpp**, in *sysname-install/Library_scheduler/include* which contains the headers for the aforementioned functions.

To use these functions, you must include the **PARALiA.hpp** header during compilation and use -lparalia (along with -Lits_path) during linking
 - Main code must be compliled/linked with C++ or nvcc.   

PARALiA BLAS functions accept usual BLAS paramaters in a similar way to OpenBLAS.
  - See *Benchmarking/PARALiA* for examples.

## Prebuild Benchmarks
 - TBD

## Related Publications:
 - CoCoPeLia: Communication-Computation Overlap Prediction for Efficient Linear Algebra on GPUs (https://ieeexplore.ieee.org/document/9408195)
 - PARALiA : A Performance Aware Runtime for Auto-tuning Linear Algebra on heterogeneous systems (https://dl.acm.org/doi/10.1145/3624569)

