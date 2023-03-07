///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation.
///

#include <cblas.h>

#include "Decomposer.hpp"
#include "unihelpers.hpp"

#include "backend_wrappers.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>


typedef struct pthread_data_in{
	void* adrs;
	long long pin_bytes;
	short pin_internally;
}* pthread_data_in_p;

void* prepareAsync_backend(void* compressed_data){
	short lvl = 2;
	pthread_data_in_p prep_data = (pthread_data_in_p)compressed_data;
	if (prep_data->pin_internally) cudaHostRegister(prep_data->adrs,prep_data->pin_bytes,cudaHostRegisterPortable);
#ifdef DEBUG
	lprintf(lvl, "pin asset= %d\n", prep_data->pin_internally);
#endif
	return NULL;
}

template<typename dtype> void Decom2D<dtype>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr){
	pthread_data_in_p prep_data = (pthread_data_in_p) malloc(sizeof(struct pthread_data_in));
	prep_data->adrs = adrs;
	// FIXME: is dim2*ldim correct in cases where part of the Asset is used in a mission (rare). Assuming col major anyway?
	prep_data->pin_bytes = dim2*ldim*dtypesize();
	cudaPointerAttributes attributes;
	// TODO: This is a cheat to understand if memory is pinned (to avoid trying to pin it if it already is)
	if (cudaSuccess!=cudaPointerGetAttributes(&attributes, prep_data->adrs)) pin_internally = 1;
	else pin_internally = 0;
	prep_data->pin_internally = pin_internally;
	cudaGetLastError();
	int s = pthread_create(thread_id, &attr, &prepareAsync_backend, (void*) prep_data);
	if (s != 0) error("Decom2D::prepareAsync: pthread_create failed s=%d\n", s);
}

template void Decom2D<double>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr);
template void Decom2D<float>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr);

template<typename dtype> void Decom2D<dtype>::resetProperties(){
	if (pin_internally) cudaHostUnregister(adrs);
}

template void Decom2D<double>::resetProperties();
template void Decom2D<float>::resetProperties();

template<typename dtype> void Decom1D<dtype>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr){
	pthread_data_in_p prep_data = (pthread_data_in_p) malloc(sizeof(struct pthread_data_in));
	prep_data->adrs = adrs;
	prep_data->pin_bytes = dim*inc*dtypesize();
	cudaPointerAttributes attributes;
	// TODO: This is a cheat to understand if memory is pinned (to avoid trying to pin it if it already is)
	if (cudaSuccess!=cudaPointerGetAttributes(&attributes, prep_data->adrs)) pin_internally = 1;
	else pin_internally = 0;
	prep_data->pin_internally = pin_internally;
	cudaGetLastError();

	int s = pthread_create(thread_id, &attr, &prepareAsync_backend, (void*) prep_data);
	if (s != 0) error("Decom2D::prepareAsync: pthread_create failed s=%d\n", s);
}

template void Decom1D<double>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr);
template void Decom1D<float>::prepareAsync(pthread_t* thread_id, pthread_attr_t attr);

template<typename dtype> void Decom1D<dtype>::resetProperties(){
	if (pin_internally) cudaHostUnregister(adrs);
}

template void Decom1D<double>::resetProperties();
template void Decom1D<float>::resetProperties();
