///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

#ifndef LOC_NUM
#define LOC_NUM (DEV_NUM + 1)
#endif

#ifndef COCONTROL_H
#define COCONTROL_H
typedef struct CoControl{
	long int T = 0; /// The tiling size used for 1D/2D Data split to tiles.
	int active_unit_num = -1; /// The number of units that will be used in the involving operation.
	int* active_unit_id_list;	/// The list of ids of said units.
	double* active_unit_score; /// The 'score' of each said units relative to the total task completion.
	double pred_t; /// The predicted seconds the whole operation will require using the above parameters.
		
	long int subkernel_num; /// The number of subkernels.
	int* Subkernels_per_dev; /// The number of subkernels derived from a unit's score that that unit unit will fire.
	int** Subkernel_dev_id_list; /// The sk_id ids of said sub-kernels, IF they are predefined and not dynamic.

	long long cache_limit = 0; /// The 'cache' size allocation limit for all devices in bytes, IF any.

}* CoControl_p;
#endif

/// create a new autotune controller (allocate/initialize it).
CoControl_p create_autotune_controller();

/// destroy an autotune controller.
CoControl_p destroy_autotune_controller();

/// Return a string represenation of the given CoControl
char* CoControlPrint(CoControl_p input);

/// Return a string with the active Cmake implemetation flag used
char* CoCoImplementationPrint();

/// Return a string with the active Cmake Subkernel Distribution flag
char* CoCoDistributionPrint();

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
CoControl_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, CoControl_p predef_control_values);

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
CoControl_p CoCopeLiaDaxpy(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDaxpyControled(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy, CoControl_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void CoCopeLiaDevCacheFree(short dev_id);

#endif
