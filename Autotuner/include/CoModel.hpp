///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The latency-bandwidth models introduced in CoCoPeLiA and extended in PARALiA for LinkMap usage.
///

#ifndef COMODEL_H
#define COMODEL_H

typedef struct  comm_data{
	long double ti;
	long double tb;
	long double sl[LOC_NUM][LOC_NUM];

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

/// Predict t_com for bytes using the link_shared_bw adjusting
double t_com_predict_shared(CoModel_p model, long double bytes);

/// Predict t_com_bid for bytes1,bytes2 using the two link CoModels
double t_com_bid_predict(CoModel_p model1, CoModel_p model2, long double bytes1, long double bytes2);

#endif
