/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the caching functions for data scheduling and management in heterogeneous multi-device systems.
///

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include<iostream>
#include <string>

//#include "unihelpers.hpp"
#include "Subkernel.hpp"

/// Each device gets 1/num_devices Subkernels without acounting for their size or location
void CoCoDistributeSubkernelsNaive(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int Subkernel_num);

/// A classic round-robin distribution
void CoCoDistributeSubkernelsRoundRobin(int* Subkernel_dev_id_list,
  int* Subkernels_per_dev, short num_devices, int Subkernel_num);

#endif
