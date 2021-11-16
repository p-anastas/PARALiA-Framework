///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cblas.h>

#include "Asset.hpp"
#include "unihelpers.hpp"

#include "backend_lib_wrappers.hpp"

typedef struct pthread_data_in{
	void* adrs;
	size_t dim1, ldim;
	short dsize, pin_internally;
}* pthread_data_in_p;

void* prepareAsync_backend(void* compressed_data){
	short lvl = 2;
	pthread_data_in_p prep_data = (pthread_data_in_p)compressed_data;
	if (prep_data->pin_internally) cudaHostRegister(prep_data->adrs,prep_data->dim1*prep_data->ldim*prep_data->dsize,cudaHostRegisterPortable);
#ifdef DEBUG
	lprintf(lvl, "pin asset= %d\n", prep_data->pin_internally);
#endif
}

template<typename dtype> void* Asset2D<dtype>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr){
	pthread_data_in_p prep_data = (pthread_data_in_p) malloc(sizeof(struct pthread_data_in));
	prep_data->adrs = adrs;
	// FIXME: is dim1*ldim correct in cases where part of the Asset is used in a mission (rare)?
	prep_data->dim1 = dim1;
	prep_data->ldim = ldim;
	prep_data->dsize = sizeof(dtype);
	cudaPointerAttributes attributes;
	// TODO: This is a cheat to understand if memory is pinned (to avoid trying to pin it if it already is)
	if (cudaSuccess!=cudaPointerGetAttributes(&attributes, prep_data->adrs)) pin_internally = 1;
	else pin_internally = 0;
	cudaGetLastError();

	int s = pthread_create(thread_id, &attr, &prepareAsync_backend, (void*) prep_data);
	if (s != 0) error("Asset2D::prepareAsync: pthread_create failed s=%d\n", s);
}

template void* Asset2D<double>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr);

template<typename dtype> void Asset2D<dtype>::resetProperties(){
	if (pin_internally) cudaHostUnregister(adrs);
}
