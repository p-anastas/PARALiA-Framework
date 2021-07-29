///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The communication sub-models used in CoCopeLia. 
///

#include <stdlib.h>
#include <math.h>

#include "unihelpers.hpp"
#include "CoCoPeLiaCoModel.hpp"

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
	sprintf(filename, "%s/Processed/Linear-Model_to-%d_from-%d.log", DEPLOYDB, to, from);
	FILE* fp = fopen(filename,"r");
	if (!fp) error("CoModel_init: t_comm LogFile =%s not generated\n",filename);
#ifdef DEBUG
	lprintf(lvl, "Reading Linear Model from %s\n", filename);
#endif
	int items = fscanf(fp, "%Le\n%Le\n%Lf", &(out_model->ti), &(out_model->tb), &(out_model->sl));
	if (items != 3) error("CoModel_init: Problem in reading model\n");
	fclose(fp);
	out_model->to = to; 
	out_model->from = from; 
	out_model->machine = (char*) malloc(256);
	strcpy(out_model->machine, TESTBED); 
#ifdef TEST
	timer = csecond() - timer;
	lprintf(lvl, "Initialization time:%lf ms\n", timer*1000);
	lprintf(lvl-1, "<-----|\n"); 
#endif
#ifdef DEBUG
	lprintf(lvl, "t_comm( %d -> %d) model initialized for %s -> ti =%Le, tb=%Le, sl = %Lf\n", out_model->from, out_model->to, out_model->machine, out_model->ti, out_model->tb, out_model->sl);
	lprintf(lvl-1, "<-----|\n"); 
#endif
	return out_model;
}


/// Predict t_com for bytes using a Cmodel 
double t_com_predict(CoModel_p model, long double bytes)
{
	if (bytes < 0) return -1;	
	else if ( bytes == 0) return 0; 
	return model->ti + model-> tb*bytes; 
}

/// Predict t_com for bytes including bidirectional use slowdown
double t_com_sl(CoModel_p model, long double bytes)
{
	if (bytes < 0) return -1;	
	else if ( bytes == 0) return 0; 
	return model->ti + model->tb*bytes*model->sl; 
}


/// Predict t_com_bid for oposing transfers of bytes1,bytes2 
double t_com_bid_predict(CoModel_p model1, CoModel_p model2, long double bytes1, long double bytes2)
{
	//return fmax(t_com_predict(model1, bytes1), t_com_predict(model2, bytes2));
	if (bytes1 < 0 || bytes2 < 0) return -1;
	else if (bytes1 == 0) return t_com_predict(model2, bytes2);
	else if (bytes2 == 0) return t_com_predict(model1, bytes1);
	double t_sl1 = t_com_sl(model1,bytes1), t_sl2 = t_com_sl(model2,bytes2);
	if (t_sl1 < t_sl2) return t_sl1*( 1.0 - 1/model2->sl) + bytes2 * model2->tb + model2->ti/model2->sl; 
	else return t_sl2*( 1.0 - 1/model1->sl) + bytes1 * model1->tb + model1->ti/model1->sl; 
}
