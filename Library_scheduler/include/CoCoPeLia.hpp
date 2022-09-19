///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

#ifndef ATC_H
#define ATC_H
typedef class ATC{
	public:
		long int T; /// The tiling size used for 1D/2D Data split to tiles.
		int active_unit_num; /// The number of units that will be used in the involving operation.
		int* active_unit_id_list;	/// The list of ids of said units.
		double* active_unit_score; /// The 'score' of each said units relative to the total task completion.
		double pred_t; /// The predicted seconds the whole operation will require using the above parameters.

		long int subkernel_num; /// The number of subkernels.
		int* Subkernels_per_unit_num; /// The number of subkernels derived from a unit's score that that unit unit will fire.
		int** Subkernels_per_unit_list; /// The sk_id ids of said sub-kernels, IF they are predefined and not dynamic.

		long long cache_limit; /// The 'cache' size allocation limit for all devices in bytes, IF any.

	ATC();
	~ATC();

	void update_sk_num(long long int subkernel_num_in);
	/// Print the basic characteristics of the autotune controller to a string
	char* print();

	/// Print the basic characteristics of the autotune controller to a string in csv-friendly format (X,Y,...)
	char* print_csv();

	/// Copy all characteristics of another utotune controller
	void mimic_ATC(class ATC* other_ATC);

}* ATC_p;
#endif

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
ATC_p CoCopeLiaDgemm(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p CoCopeLiaDgemmControled(char TransA,  char TransB, size_t M, size_t N, size_t K,
	double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, ATC_p predef_control_values);

/// The CoCopeLia Dgemm implementation. A prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
ATC_p CoCopeLiaDaxpy(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy);

/// A modification of CoCopeLiaDgemm but with a given T (mainly for performance/debug purposes)
ATC_p CoCopeLiaDaxpyControled(size_t N, VALUE_TYPE alpha,
	VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy, ATC_p predef_control_values);

///Deallocates the GPU-allocated cache buffer at target device
void CoCopeLiaDevCacheFree(short dev_id);

#endif
