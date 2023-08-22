///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include <cassert>
#include <stdlib.h>

#include "Autotuning_runtime.hpp"
#include "Werkhoven.hpp"
#include "linkmap.hpp"

int main(const int argc, const char *argv[]) {

  	int ctr = 1;

	char machine[256], func[256], filename[256];
	size_t Dim1 = 1, Dim2 = 1, Dim3 = 1, offset1 = 1, offset2 = 1, offset3 = 1;

	int dev_id = -1; 

	char flag1 = 'X', flag2 = 'X', flag3 = 'X'; 
	short Loc1 = 1, Loc2 = 1, Loc3 = 1, OutLoc1 = 1, OutLoc2 = 1, OutLoc3 = 1; 
	switch (argc) {
	case (18):
		dev_id = atoi(argv[ctr++]);
		sprintf(func , "%s", argv[ctr++]);

		flag1 = argv[ctr++][0];
		flag2 = argv[ctr++][0];
		flag3 = argv[ctr++][0];

		Dim1 = atoi(argv[ctr++]);
		Dim2 = atoi(argv[ctr++]);
		Dim3 = atoi(argv[ctr++]);

		Loc1 = atoi(argv[ctr++]);
		Loc2 = atoi(argv[ctr++]);	
		Loc3 = atoi(argv[ctr++]);

		OutLoc1 = atoi(argv[ctr++]);
		OutLoc2 = atoi(argv[ctr++]);
		OutLoc3 = atoi(argv[ctr++]);

		offset1 = atoi(argv[ctr++]);
		offset2 = atoi(argv[ctr++]);	
		offset3 = atoi(argv[ctr++]);

		break;

	default:
		error("Incorrect input arguments. Usage: ./correct_run dev_id func(={Dgemm,Sgemm}) flag1 flag2 flag3 Dim1 Dim2 Dim3 Loc1 Loc2 Loc3 OutLoc1 OutLoc2 OutLoc3 offset1 offset2 offset3");
  	}

	sprintf(filename, "%s/predictions/CoCopeLiaLogPrediction-%s_dev-%d.log", TESTDIR, func, dev_id);
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	CoCoModel_p CoComodel = CoCoPeLiaModelInit(dev_id, func, flag1, flag2, flag3, Dim1, Dim2, Dim3, Loc1, Loc2, Loc3, OutLoc1, OutLoc2, OutLoc3, offset1, offset2, offset3);

	double timer;
	for (short modeIdx = 0; modeIdx < COCOPELIA_PIPELINE_EMULATE + 1; modeIdx++){
		ModelType mode = (ModelType) modeIdx; 
		fprintf(stderr,"%s\n", printModel(mode)); 

		timer = csecond(); 
		tunableParams_p coco = CoCoPeLiaModelOptimizeTile(CoComodel, mode);
		timer = csecond() - timer;
		fprintf(stderr, "Prediction time: %lf ms\n", timer);	
		fprintf(fp,"%s, %d, %s, %c,%c,%c, %zu,%zu,%zu, %d,%d,%d, %d,%d,%d, %zu,%zu,%zu, %s, %e\n", printModel(mode), dev_id, func, flag1, flag2, flag3, Dim1, Dim2, Dim3, Loc1, Loc2, Loc3, OutLoc1, OutLoc2, OutLoc3, offset1, offset2, offset3, printTunableParams(coco), timer);
	}
        fclose(fp); 
	free(CoComodel);
	return 0;
}
