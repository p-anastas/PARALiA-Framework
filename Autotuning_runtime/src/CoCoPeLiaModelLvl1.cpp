///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS1
///

#include <stdlib.h>
#include <math.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "CoCoPeLiaModel.hpp"
#include "unihelpers.hpp"
#include "Werkhoven.hpp"


///  Initializes the model for gemm
CoCoModel_p CoCoModel_axpy_init(CoCoModel_p out_model, short dev_id, const char* func, axpy_backend_in_p func_data){
	long int N = func_data->N;
	short x_loc, x_out_loc = x_loc = CoCoGetPtrLoc(*func_data->x),
				y_loc, y_out_loc = y_loc = CoCoGetPtrLoc(*func_data->y);
	long int incx = func_data->incx, incy = func_data->incy;
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoModel_axpy_init(model,%ld, %d, %d, %d, %d, %d, %d, %d, %s)\n",
		N, x_loc, y_loc, x_out_loc, y_out_loc, incx, incy, dev_id, func);
#endif
	out_model->func = func;
	// Axpy Routine info
	out_model->V->numT = 2;

	if (!strcmp(func, "Daxpy")) out_model->V->dtype_sz = sizeof(double);
	else if (!strcmp(func, "Saxpy")) out_model->V->dtype_sz = sizeof(float);

	out_model->V->in[0] = 1;
	out_model->V->in[1] = 1;

	out_model->V->out[0] = 0;
	out_model->V->out[1] = 1;

	out_model->D1 = N;
	out_model->D2 = -1;
	out_model->D3 = -1;

	out_model->V->Dim1[0] = &out_model->D1;
	out_model->V->Dim1[1] = &out_model->D1;

	out_model->V->Dim2[0] = NULL;
	out_model->V->Dim2[1] = NULL;

	out_model->V->loc[0] = x_loc;
	out_model->V->loc[1] = y_loc;

	out_model->V->out_loc[0] = x_out_loc;
	out_model->V->out_loc[1] = y_out_loc;

#ifdef DEBUG
	lprintf(lvl, "CoCoModel_axpy initalized for %s->\nInitial problem dims: D1 = %ld, D2 = %ld, D3 = %ld\n"
	"Data tiles : x(%ld), y(%ld), in loc (%d,%d)\n", \
	func, out_model->D1, out_model->D2, out_model->D3, out_model->D1, out_model->D1, out_model->V->out_loc[0], out_model->V->out_loc[1]);
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
}

///  Initializes the model for gemm
CoCoModel_p CoCoModelFuncInitBLAS1(CoCoModel_p out_model, short dev_id, const char* func, void* func_data){
	if ( !strcmp(func, "Daxpy") || !strcmp(func, "Saxpy"))
		return CoCoModel_axpy_init(out_model, dev_id, func, (axpy_backend_in_p) func_data);
	else error("CoCoModelFuncInitBLAS1: func %s not implemented\n", func);
}
