///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cblas.h>

#include "backend_lib_wrappers.hpp"

int CoCoPeLiaBackendGetDevID();
void CoCoPeLiaBackendSetDevID(int devID);
void CoCoPeLiaBackendSync();

