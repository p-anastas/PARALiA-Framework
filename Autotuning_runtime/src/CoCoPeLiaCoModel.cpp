///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The communication sub-models used in CoCopeLia.
///

#include <stdlib.h>
#include <math.h>

#include "unihelpers.hpp"
#include "CoCoPeLiaCoModel.hpp"
#include "Autotuning_runtime.hpp"

CoModel_p CoModel_init(short to, short from)
{
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoModel_init(to=%d,from=%d)\n", to, from);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoModel_init\n");
	double timer = csecond();
#endif
	CoModel_p out_model = (CoModel_p) malloc(sizeof(struct  comm_data));
	char filename[256];
	sprintf(filename, "%s/Processed/Link-Stats_loc_dest-%d_loc_src-%d.log", DEPLOYDB, to, from);
	FILE* fp = fopen(filename,"r");
	if (!fp) error("CoModel_init(%d,%d): t_comm LogFile =%s not generated\n", to, from, filename);
#ifdef DPDEBUG
	lprintf(lvl, "Reading Linear Model from %s\n", filename);
#endif
	int items = fscanf(fp, "Intercept: %Le\nCoefficient: %Le\n", &(out_model->ti), &(out_model->tb));
	if (items != 2) error("CoModel_init: Problem in reading model Inter/Coef\n");
	for(int d1 = 0; d1 < LOC_NUM; d1++) for(int d2 = 0; d2 < LOC_NUM; d2++) out_model->sl[d1][d2] = 1;
	int loc_dest, loc_src;
	long double sl;
	while(!feof(fp)){
		items = fscanf(fp, "Slowdown([%d]->[%d]): %Le\n", &(loc_src), &(loc_dest), &(sl));
		if (items != 3) error("CoModel_init(%d,%d): Problem in reading model Slowdown\n", to, from);
		if (sl < 1 ){
#ifdef DPDEBUG
	warning("CoModel_init( %d -> %d): sl(%d -> %d) = %Lf, check\n",
		from, to, loc_src, loc_dest, sl);
#endif
			sl = 1;
		}
		else if (sl > 2){
#ifdef DPDEBUG
			warning("CoModel_init( %d -> %d): sl(%d -> %d) = %Lf, reseting to 2.0 for 50%% bw sharing\n",
				from, to, loc_src, loc_dest, sl);
#endif
			sl = 2.0;
		}
		if (!(abs(sl - 1) < NORMALIZE_NEAR_SPLIT_LIMIT)) out_model->sl[idxize(loc_dest)][idxize(loc_src)] = sl;
	}
	out_model->to = to;
	out_model->from = from;
	out_model->machine = (char*) malloc(256*sizeof(char));
	strcpy(out_model->machine, TESTBED);
#ifdef PDEBUG
	lprintf(lvl, "t_comm( %d -> %d) model initialized for %s -> ti =%Le, tb=%Le\n", out_model->from, out_model->to, out_model->machine, out_model->ti, out_model->tb);
#endif
	fclose(fp);
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Initialization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
}

CoModel_p CoModel_init_local(short dev_id)
{
	short lvl = 3;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoModel_init_local(dev_id=%d)\n", dev_id);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoModel_init_local\n");
	double timer = csecond();
#endif
	CoModel_p out_model = (CoModel_p) malloc(sizeof(struct  comm_data));
	out_model->to = dev_id;
	out_model->from = dev_id;
	out_model->machine = (char*) malloc(256*sizeof(char));
	strcpy(out_model->machine, TESTBED);
	out_model->ti = 0.0;
	out_model->tb = 0.0;
	for(int d1 = 0; d1 < LOC_NUM; d1++) for(int d2 = 0; d2 < LOC_NUM; d2++) out_model->sl[d1][d2] = 1;
#ifdef PDEBUG
	lprintf(lvl, "t_comm( %d -> %d) model initialized for %s -> ti =%Le, tb=%Le\n", out_model->from, out_model->to, out_model->machine, out_model->ti, out_model->tb);
#endif
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Initialization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return out_model;
}


/// Predict t_com for bytes using a Cmodel
double t_com_predict(CoModel_p model, long double bytes)
{
	if (bytes < 0) return -1;
	else if ( bytes == 0) return 0;
#ifdef DPDEBUG
		lprintf(4, "t_com_predict(%Lf): ti = %Lf, tb = %Lf -> t_com = %Lf ms\n",
			bytes, model->ti, model-> tb, (model->ti + model-> tb*bytes)*1000);
#endif
	return model->ti + model-> tb*bytes;
}

/// Predict t_com for bytes using a Cmodel
double t_com_predict_shared(CoModel_p model, long double bytes)
{
	if (bytes < 0) return -1;
	else if ( bytes == 0) return 0;
#ifdef ENABLE_TRANSFER_HOPS
	else if (link_shared_bw_hop[idxize(model->to)][idxize(model->from)] == 0.0) return 0;
#else
	else if (link_shared_bw[idxize(model->to)][idxize(model->from)] == 0.0) return 0;
#endif	
#ifdef DPDEBUG
		lprintf(4, "t_com_predict_shared(%Lf): ti = %Lf, tb = %Lf, link_bw[%d][%d] = %lf, link_shared_bw[%d][%d] = %lf-> t_com = %Lf ms\n",
			bytes, model->ti, model-> tb, (model->ti + model-> tb*bytes)*1000, model->to, model->from, link_bw[idxize(model->to)][idxize(model->from)],
		model->to, model->from, link_shared_bw[idxize(model->to)][idxize(model->from)]);
#endif
#ifdef ENABLE_TRANSFER_HOPS
	return (link_bw[idxize(model->to)][idxize(model->from)]/
		link_shared_bw_hop[idxize(model->to)][idxize(model->from)])*
		(model->ti + model-> tb*bytes);
#else
	return (link_bw[idxize(model->to)][idxize(model->from)]/
		link_shared_bw[idxize(model->to)][idxize(model->from)])*
		(model->ti + model-> tb*bytes);
#endif
}

/// Predict t_com for bytes including bidirectional use slowdown
double t_com_sl(CoModel_p model, long double bytes)
{
	if (bytes < 0) return -1;
	else if ( bytes == 0) return 0;
	return model->ti + model->tb*bytes*model->sl[idxize(model->from)][idxize(model->to)];
}


/// Predict t_com_bid for oposing transfers of bytes1,bytes2
double t_com_bid_predict(CoModel_p model1, CoModel_p model2, long double bytes1, long double bytes2)
{
	//return fmax(t_com_predict(model1, bytes1), t_com_predict(model2, bytes2));
	if (bytes1 < 0 || bytes2 < 0) return -1;
	else if (bytes1 == 0) return t_com_predict(model2, bytes2);
	else if (bytes2 == 0) return t_com_predict(model1, bytes1);
	double t_sl1 = t_com_sl(model1,bytes1), t_sl2 = t_com_sl(model2,bytes2);
	if (t_sl1 < t_sl2) return t_sl1*( 1.0 - 1/model2->sl[idxize(model2->from)][idxize(model2->to)]) +
		bytes2 * model2->tb + model2->ti/model2->sl[idxize(model2->from)][idxize(model2->to)];
	else return t_sl2*( 1.0 - 1/model1->sl[idxize(model1->from)][idxize(model1->to)]) +
		bytes1 * model1->tb + model1->ti/model1->sl[idxize(model1->from)][idxize(model1->to)];
}
