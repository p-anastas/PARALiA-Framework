///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef WERKHOVEN_H
#define WERKHOVEN_H

typedef struct werkhoven_model{	
	CoModel_p h2d;
	CoModel_p d2h;
	// FIXME: model for kernel execution not explored py paper. Use the same with proposed in order to compare final models with similar 2nd order errors. 
	void* GPUexec_model_ptr;
}* WerkhovenModel_p; 

/// Initializes underlying models required for Werkhoven
WerkhovenModel_p WerkhovenModel_init(short dev_id, char* func, short level , short mode);

///  Predicts 3-way overlaped execution time for nStream (equal) data blocks of any kernel using Werkhoven's model. 
double WerkhovenModel_predict(WerkhovenModel_p model, long long h2d_bytes, long long d2h_bytes, double nStreams, short level, size_t D1,  size_t D2, size_t D3);

#endif
