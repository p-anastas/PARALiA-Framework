///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_COMODEL_H
#define COCOPELIA_COMODEL_H

typedef struct  comm_data{
	long double ti;
	long double tb;
	long double sl;

	short from, to;
	char* machine;
}* CoModel_p;

/// Load parameters from file and return model
CoModel_p CoModel_init(short to, short from);

/// Create a dummy model in dev_id (for no transfer-medium cases)
CoModel_p CoModel_init_local(short dev_id);

/// Predict t_com for bytes using the CoModel
double t_com_predict(CoModel_p model, long double bytes);

/// Predict t_com for bytes using the CoModel with bidirectional overlap slowdown
double t_com_sl(CoModel_p model, long double bytes);

/// Predict t_com_bid for bytes1,bytes2 using the two link CoModels
double t_com_bid_predict(CoModel_p model1, CoModel_p model2, long double bytes1, long double bytes2);

#define maxDim_trans 16384
#define minDim_trans 256
#define step_trans 256

typedef struct  comodel{
	short from, to;

	// 0 for undef, 1 for average, TODO: 2 for min, 3 for max
	short mode;
	double av_time_buffer_tile[(maxDim_trans-minDim_trans)/step_trans + 1];
	// TODO: These can be used for more robust results or for worst/best case performance prediction
	double min_time_buffer_tile[(maxDim_trans-minDim_trans)/step_trans + 1];
	double max_time_buffer_tile[(maxDim_trans-minDim_trans)/step_trans + 1];

	double av_time_sl_buffer_tile[(maxDim_trans-minDim_trans)/step_trans + 1];
	// TODO: These can be used for more robust results or for worst/best case performance prediction
	double min_time_sl_buffer_tile[(maxDim_trans-minDim_trans)/step_trans + 1];
	double max_time_sl_buffer_tile[(maxDim_trans-minDim_trans)/step_trans + 1];

	double av_time_buffer_vec[(maxDim_trans-minDim_trans)/step_trans + 1];
	// TODO: These can be used for more robust results or for worst/best case performance prediction
	double min_time_buffer_vec[(maxDim_trans-minDim_trans)/step_trans + 1];
	double max_time_buffer_vec[(maxDim_trans-minDim_trans)/step_trans + 1];

	double av_time_sl_buffer_vec[(maxDim_trans-minDim_trans)/step_trans + 1];
	// TODO: These can be used for more robust results or for worst/best case performance prediction
	double min_time_sl_buffer_vec[(maxDim_trans-minDim_trans)/step_trans + 1];
	double max_time_sl_buffer_vec[(maxDim_trans-minDim_trans)/step_trans + 1];

}* ComModel_p;

/// Load parameters from file and return model
ComModel_p ComModel_init(short to, short from, short mode);

/// Predict t_comm for a tile of size TxT of dtype
double CoTile_predict(ComModel_p model, size_t T, short dtype_sz);

/// Predict t_comm for a vec of size 1xT of dtype
double CoVec_predict(ComModel_p model, size_t T, short dtype_sz);

/// Predict t_comm for bytes using the CoModel with bidirectional overlap slowdown
double CoTile_sl_predict(ComModel_p model, size_t T, short dtype_sz);

double CoTile_bid_predict(ComModel_p model_h2d, ComModel_p model_d2h, size_t T, short dtype_sz, short numTin, short numTout);

#endif
