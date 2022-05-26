///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The model from "Performance models for CPU-GPU data transfers"
///
/// Werkhoven:
/// Overlap model for no implicit synchronization and 2 copy engines:
/// time = max(tThd + tE/nStreams + tTdh/nStreams,
/// tThd/nStreams + tE + tTdh/nStreams,
/// tThd/nStreams + tE/nStreams + tTdh)
/// Single Data transfer: LogP
/// Large transfer with multiple streams LogGP
/// Total model for streaming overlap with nStreams blocks
/// time = max(L + o + B_sent*Gh2d + g*(nStreams − 1) + tE/nStreams + L + o + B_recv*Gd2h/nStreams + g*(nStreams − 1),
/// L + o + B_recv*Gh2d/nStreams + g*(nStreams − 1) + tE + L + o + B_recv*Gd2h/nStreams + g*(nStreams − 1),
/// L + o + B_sent*Gh2d/nStreams + g*(nStreams − 1) + tE/nStreams + L + o + B_recv*Gd2h + g*(nStreams − 1))
/// TODO: We use for this model g = L + o since the actual g is very hard to calculate accurately (same as werkhoven). Since g <= L + o, this results in a small time overestimation,
///	  in our case this is not a problem since Werkhoven always underestimates time because of its linearily.

#include <stdlib.h>
#include <math.h>

#include "CoCoPeLiaCoModel.hpp"
#include "CoCoPeLiaGPUexec.hpp"
#include "CoCoPeLiaModel.hpp"
#include "unihelpers.hpp"

///  Predicts 3-way overlaped execution time for nStream (equal) data blocks of any kernel using Werkhoven's model.
double WerkhovenModelInternal(CoCoModel_p model, long long h2d_bytes, long long d2h_bytes, double nStreams, double kernel_time_0, double kernel_time_1)
{
	if (!nStreams) return -1.0;
	/// T[0] = Serialized/Dominant, T[1] = Blocked with nStreams
	double serial_time, h2d_dom_time, ker_dom_time, d2h_dom_time, result, d2h_time[2], h2d_time[2];

	if ( kernel_time_0 < 0 ){
		warning("WerkhovenModelInternal: GPUexec3Model_predict submodel returned negative value, abort prediction");
		return -1.0;
	}

	CoModel_p h2d_model = model->link[DEV_NUM-1], d2h_model = model->revlink[DEV_NUM-1];
	h2d_time[0] = h2d_model->ti + h2d_model->tb*h2d_bytes + h2d_model->ti*(nStreams - 1);
	h2d_time[1] = h2d_model->ti + h2d_model->tb*h2d_bytes/nStreams;
	d2h_time[0] = d2h_model->ti + d2h_model->tb*d2h_bytes + d2h_model->ti*(nStreams - 1);
	d2h_time[1] = d2h_model->ti + d2h_model->tb*d2h_bytes/nStreams;

	h2d_dom_time = h2d_time[0] + kernel_time_1 + d2h_time[1];
	ker_dom_time = h2d_time[1] + kernel_time_0 + d2h_time[1];
	d2h_dom_time = h2d_time[1] + kernel_time_1 + d2h_time[0];
	serial_time = h2d_model->ti + h2d_model->tb*h2d_bytes + kernel_time_0 +  d2h_model->ti + d2h_model->tb*d2h_bytes;

	result =  fmax(fmax(h2d_dom_time, ker_dom_time), d2h_dom_time);

	fprintf(stderr, "Werkhoven predicted ( Streams = %lf) -> ", nStreams);
	double res_h2d = h2d_time[1], res_ker = kernel_time_1, res_d2h = d2h_time[1];
	if (h2d_dom_time == result) {
		res_h2d = h2d_time[0];
		fprintf(stderr, "H2D dominant problem\n");
	}
	if (ker_dom_time == result) {
		res_ker = kernel_time_0;
		fprintf(stderr, "Exec dominant problem\n");
	}
	if (d2h_dom_time == result){
		res_d2h = d2h_time[0];
		fprintf(stderr, "D2h dominant problem\n");
	}

	double overlap_speedup = ( serial_time - result) / serial_time;

	fprintf(stderr, "\tt_h2d: %lf ms\n"
	"\tt_exec: %lf ms\n"
	"\tt_d2h: %lf ms\n"
	"\t -> total: %lf ms (%lf GFlops/s)\n"
	"\tExpected Speedup = \t%.3lf\n\n",
	1000*res_h2d, 1000*res_ker, 1000*res_d2h, 1000*result, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), result), overlap_speedup);

	return result;
}


//FIXME: Fails for some tiles because of adaptive step
///  Predicts 3-way overlaped execution time for nStream (equal) data blocks of any kernel using Werkhoven's model.
double WerkhovenModelPredict(CoCoModel_p model, long long h2d_bytes, long long d2h_bytes, long int T)
{
	// Linear performance assumption used by werkhoven.
	double t_kernel = 0;
	long int maxT = GPUexec3MaxT((GPUexec3Model_p)model->GPUexec_model_ptr); //CoCopeLiaGetMaxSqdimLvl3(model->V->numT, model->V->dtype_sz, STEP_BLAS3); //
	long int Tbig = GPUexec3NearestT((GPUexec3Model_p)model->GPUexec_model_ptr, fmin(maxT, fmin(fmin(model->D1,model->D2), model->D3)));
	//fprintf(stderr, "Tbig = %zu\n", Tbig);
	t_kernel = (model->D1*1.0/Tbig * model->D2*1.0/Tbig * model->D3*1.0/Tbig)* GPUexec3Model_predict((GPUexec3Model_p)model->GPUexec_model_ptr, Tbig, model->flags->TransA, model->flags->TransB);
	double nStreams = (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T);
	return WerkhovenModelInternal(model, h2d_bytes, d2h_bytes, nStreams, t_kernel, t_kernel/nStreams);
}

///  Predicts 3-way overlaped execution time for nStream (equal) data blocks of any kernel using Werkhoven's model but with the tile exec time (bottom-up).
double WerkhovenModelPredictExecTiled(CoCoModel_p model, long long h2d_bytes, long long d2h_bytes, long int T)
{
	// Modified version which uses exec time lookup like CoCopeLia.
	double t_kernel = 0;
	t_kernel =  GPUexec3Model_predict((GPUexec3Model_p)model->GPUexec_model_ptr, T, model->flags->TransA, model->flags->TransB);
	double nStreams = (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T);
	return WerkhovenModelInternal(model, h2d_bytes, d2h_bytes, nStreams, t_kernel*nStreams, t_kernel);
}

double WerkhovenModelPredictWrapper(CoCo_model* model, long int T, short t_exec_method){ // t_exec_method: 0 = default (top down), 1 = Data-loc aware, 2 = Tile-specific lookup
	if (!strcmp(model->func, "Dgemm")) switch(t_exec_method){
			case 0: return WerkhovenModelPredict(model, (model->D1*model->D3 + model->D3*model->D2 + model->D1*model->D2)*sizeof(double), model->D1*model->D2*sizeof(double), T);
			case 1: return WerkhovenModelPredict(model, (model->V->out_loc[0]*model->D1*model->D3 + model->V->out_loc[1]*model->D3*model->D2 + model->V->out_loc[2]*model->D1*model->D2)*sizeof(double), model->V->out_loc[2]*model->D1*model->D2*sizeof(double), T);
			case 2: return WerkhovenModelPredictExecTiled(model, (model->V->out_loc[0]*model->D1*model->D3 + model->V->out_loc[1]*model->D3*model->D2 + model->V->out_loc[2]*model->D1*model->D2)*sizeof(double), model->V->out_loc[2]*model->D1*model->D2*sizeof(double), T);
			default: error("WerkhovenModelPredictWrapper: invalid t_exec_method %d", t_exec_method);
		}
	else if (!strcmp(model->func, "Sgemm")) switch(t_exec_method){
			case 0: return WerkhovenModelPredict(model, (model->D1*model->D3 + model->D3*model->D2 + model->D1*model->D2)*sizeof(float), model->D1*model->D2*sizeof(float), T);
			case 1: return WerkhovenModelPredict(model, (model->V->out_loc[0]*model->D1*model->D3 + model->V->out_loc[1]*model->D3*model->D2 + model->V->out_loc[2]*model->D1*model->D2)*sizeof(float), model->V->out_loc[2]*model->D1*model->D2*sizeof(float), T);
			case 2: return WerkhovenModelPredictExecTiled(model, (model->V->out_loc[0]*model->D1*model->D3 + model->V->out_loc[1]*model->D3*model->D2 + model->V->out_loc[2]*model->D1*model->D2)*sizeof(float), model->V->out_loc[2]*model->D1*model->D2*sizeof(float), T);
			default: error("WerkhovenModelPredictWrapper: invalid t_exec_method %d", t_exec_method);
		}
	//else if (!strcmp(func, "Dgemv")) error("WerkhovenModelPredictWrapper: func '%s' model not implemented", func); return WerkhovenModel_predict(model, first_loc*D1*D2*sizeof(double) + second_loc*D2*sizeof(double) + third_loc*D1*sizeof(double), third_loc*D1*sizeof(double), (1.0*D1/T), 2, D1,D2,-1);
	//else if (!strcmp(func, "Daxpy")); return WerkhovenModel_predict(model, first_loc*D1*sizeof(double) + second_loc*D1*sizeof(double), second_loc*D1*sizeof(double), (1.0*D1/T), 1, D1,-1,-1);
	else error("WerkhovenModelPredictWrapper: func '%s' model not implemented", model->func);
}
