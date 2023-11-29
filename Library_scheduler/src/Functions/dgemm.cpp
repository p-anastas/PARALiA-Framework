///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The DGEMM CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "Autotuner.hpp"
#include "Decomposer.hpp"
#include "PARALiA.hpp"
#include "linkmap.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>

ATC_p predef_controller_dgemm = NULL;

#ifdef STEST
double gemm_entry_ts;
#endif

void ManageCachesDgemm(PMD_p local_PMD){
	int T = local_PMD->autotuner->T; 
	int Block_num_A = (local_PMD->decom[0]->dim1/T + ((local_PMD->decom[0]->dim1%T)? 1 : 0))
			* (local_PMD->decom[0]->dim2/T + ((local_PMD->decom[0]->dim2%T)? 1 : 0)),
		Block_num_B = (local_PMD->decom[1]->dim1/T + ((local_PMD->decom[1]->dim1%T)? 1 : 0))
			* (local_PMD->decom[1]->dim2/T + ((local_PMD->decom[1]->dim2%T)? 1 : 0)),
		Block_num_C = (local_PMD->decom[2]->dim1/T + ((local_PMD->decom[2]->dim1%T)? 1 : 0))
			* (local_PMD->decom[2]->dim2/T + ((local_PMD->decom[2]->dim2%T)? 1 : 0));
	long long Block_sz = 	T*T*sizeof(double);
	for(int cache_loc = 0; cache_loc < LOC_NUM; cache_loc++){
		int Block_num = 0, Native_block_num = 0;
		if (local_PMD->decom[0]->loc == deidxize(cache_loc)) Native_block_num+=Block_num_A;
		if (local_PMD->decom[1]->loc == deidxize(cache_loc)) Native_block_num+=Block_num_B;
		if (local_PMD->decom[2]->loc == deidxize(cache_loc)) Native_block_num+=Block_num_C;

		long long max_cache_sz = 0;
		if(local_PMD->autotuner->cache_limit > 0) {
			max_cache_sz = local_PMD->autotuner->cache_limit;
			if (max_cache_sz < 3 * Block_sz) 
				error("PARALiADgemm: Problem cannot run with less memory than %d\n", 3 * Block_sz);

			long long free_dev_mem, max_dev_mem = 0, prev_DevCache_sz = 0;
			if (current_SAB[cache_loc] != NULL) prev_DevCache_sz = (long long)
				current_SAB[cache_loc]->BlockSize* current_SAB[cache_loc]->BlockNum;
			int prev_dev = CoCoPeLiaGetDevice();
			CoCoPeLiaSelectDevice(deidxize(cache_loc));

			if(deidxize(cache_loc)!=-1) {
			// if(Native_block_num==0){
				CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
				max_cache_sz = (long long) fmin(max_cache_sz, free_dev_mem 
					- ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz);
			}
			else {
				// free_dev_mem = max_dev_mem = 2 * Native_block_num * Block_sz;
				free_dev_mem = max_dev_mem = (Native_block_num + 3) * Block_sz;
				max_cache_sz = free_dev_mem;
			}
			max_cache_sz = fmax((Native_block_num + 3) * Block_sz, max_cache_sz);

			CoCoPeLiaSelectDevice(prev_dev);
			// max_cache_sz = (long long) fmin(max_cache_sz, free_dev_mem 
			// - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz);
		}
		else{
			long long free_dev_mem, max_dev_mem = 0, prev_DevCache_sz = 0;
			if (current_SAB[cache_loc] != NULL) prev_DevCache_sz = (long long)
				current_SAB[cache_loc]->BlockSize* current_SAB[cache_loc]->BlockNum;
			int prev_dev = CoCoPeLiaGetDevice();
			CoCoPeLiaSelectDevice(deidxize(cache_loc));

			if(deidxize(cache_loc)!=-1) CoCoPeLiaDevGetMemInfo(&free_dev_mem, &max_dev_mem);
			// TODO: hard coded value, should put something that reads it from system?
			else free_dev_mem = max_dev_mem = 100000000000; 
			CoCoPeLiaSelectDevice(prev_dev);
			max_cache_sz = free_dev_mem - ((long long) max_dev_mem*(1-PROBLEM_GPU_PERCENTAGE/100.0)) + prev_DevCache_sz;
		}
		Block_num = 1 + Block_num_A + Block_num_B + Block_num_C;
		if (!strcmp(OUTPUT_ALGO_MODE,"ALGO_WR_LAZY") || !strcmp(OUTPUT_ALGO_MODE,"ALGO_WREDUCE")) Block_num+= Block_num_C; 
		int max_block_num = max_cache_sz/Block_sz;
		if(max_block_num < Block_num){
			lprintf(0, "PARALiADgemm: Problem will use %d blocks for dev_id = %d\
				instead of %d needed for the full problem\n", max_block_num, deidxize(cache_loc), Block_num);
			lprintf(0, "====================================\n");
			Block_num = max_block_num;
			// 2 + (local_PMD->decom[2]->dim1/T + ((local_PMD->decom[2]->dim1%T)? 1 : 0))
			// * (local_PMD->decom[2]->dim2/T + ((local_PMD->decom[2]->dim2%T)? 1 : 0));
			int worst_case_ex_blocks = 3; 
			if(max_block_num < worst_case_ex_blocks)
				error("PARALiADgemm: Not able to run with < %d blocks per cache due to EX scheduling\n", 
					worst_case_ex_blocks);
		}
#ifdef BUFFER_REUSE_ENABLE
		if(current_SAB[cache_loc] == NULL) current_SAB[cache_loc] = 
			new Buffer(deidxize(cache_loc), Block_num, Block_sz);
		else if (current_SAB[cache_loc]->BlockSize != Block_sz 
			|| current_SAB[cache_loc]->BlockNum < Block_num){
#ifdef DEBUG
			fprintf(stderr, "PARALiADgemm: Previous Cache smaller than requested:\
			current_SAB[%d]->BlockSize=%lld vs Block_sz = %lld,\
			current_SAB[%d]->BlockNum=%d vs Block_num = %d\n",
			cache_loc, current_SAB[cache_loc]->BlockSize, Block_sz,
			cache_loc, current_SAB[cache_loc]->BlockNum, Block_num);
#endif
			delete current_SAB[cache_loc];
			current_SAB[cache_loc] = new Buffer(deidxize(cache_loc), Block_num, Block_sz);
		}
		else{
			;
		}
#else
			if(current_SAB[cache_loc]!= NULL) 
				error("PARALiADgemm: current_SAB[%d] was not NULL with reuse disabled\n", cache_loc);
			current_SAB[cache_loc] = new Buffer(deidxize(cache_loc), Block_num, Block_sz);
#endif
	}
	for (int i = 0; i < LOC_NUM; i++) local_PMD->SAB[i] = current_SAB[i];
	return;
}

void CreateSubkernelsDgemm(PMD_p local_PMD){
	/// Check if decomposers satisfy GEMM dim criteria for N, N transpose
	if (local_PMD->decom[0]->transpose == 'N' && local_PMD->decom[1]->transpose == 'N'){
		massert(local_PMD->decom[0]->GridSz1 == local_PMD->decom[2]->GridSz1 &&
						local_PMD->decom[0]->Tile_map[0]->dim1 == local_PMD->decom[2]->Tile_map[0]->dim1 &&
						local_PMD->decom[0]->Tile_map[local_PMD->decom[0]->GridSz1*local_PMD->decom[0]->GridSz2-1]->dim1
						== local_PMD->decom[2]->Tile_map[local_PMD->decom[2]->GridSz1*local_PMD->decom[2]->GridSz2-1]->dim1,
						"M dim does not mach between decomposers for GEMM\n");
		massert(local_PMD->decom[1]->GridSz2 == local_PMD->decom[2]->GridSz2 &&
						local_PMD->decom[1]->Tile_map[0]->dim2 == local_PMD->decom[2]->Tile_map[0]->dim2 &&
						local_PMD->decom[1]->Tile_map[local_PMD->decom[1]->GridSz1*local_PMD->decom[1]->GridSz2-1]->dim2
						== local_PMD->decom[2]->Tile_map[local_PMD->decom[2]->GridSz1*local_PMD->decom[2]->GridSz2-1]->dim2,
						"N dim does not mach between decomposers for GEMM\n");
		massert(local_PMD->decom[0]->GridSz2 == local_PMD->decom[1]->GridSz1 &&
						local_PMD->decom[0]->Tile_map[0]->dim2 == local_PMD->decom[1]->Tile_map[0]->dim1 &&
						local_PMD->decom[0]->Tile_map[local_PMD->decom[0]->GridSz1*local_PMD->decom[0]->GridSz2-1]->dim2
						== local_PMD->decom[1]->Tile_map[local_PMD->decom[1]->GridSz1*local_PMD->decom[1]->GridSz2-1]->dim1,
						"K dim does not mach between decomposers for GEMM\n");
	}
	local_PMD->sk_num = local_PMD->autotuner->subkernel_num;
#ifdef DEBUG
	fprintf(stderr, "|-----> CreateSubkernelsDgemm(%p,%d,%d)\n",
		local_PMD, local_PMD->autotuner->T, local_PMD->sk_num);
	fprintf(stderr,"MgridSz = %d, NgridSz = %d, KgridSz = %d\n",
		local_PMD->decom[0]->GridSz1, local_PMD->decom[1]->GridSz2, local_PMD->decom[0]->GridSz2);
	fprintf(stderr,"Mlast = %d, Nlast = %d, Klast = %d\n",
	local_PMD->decom[0]->Tile_map[local_PMD->decom[0]->GridSz1*local_PMD->decom[0]->GridSz2-1]->dim1,
	local_PMD->decom[1]->Tile_map[local_PMD->decom[1]->GridSz1*local_PMD->decom[1]->GridSz2-1]->dim2,
	local_PMD->decom[0]->Tile_map[local_PMD->decom[0]->GridSz1*local_PMD->decom[0]->GridSz2-1]->dim2);
#endif

	local_PMD->subkernel_list = (Subkernel**) malloc(local_PMD->sk_num*sizeof(Subkernel*));
	gemm_backend_in<double>* initial_dgemm = (gemm_backend_in<double>*) local_PMD->problem_wrap;
	int current_ctr = 0;
	for (int mi = 0; mi < local_PMD->decom[0]->GridSz1; mi++){
		for (int ni = 0; ni < local_PMD->decom[1]->GridSz2; ni++){
			for (int ki = 0; ki < local_PMD->decom[0]->GridSz2; ki++){
	      		current_ctr = mi*local_PMD->decom[1]->GridSz2*local_PMD->decom[0]->GridSz2 + 
					ni*local_PMD->decom[0]->GridSz2 + ki;
				local_PMD->subkernel_list[current_ctr] = new Subkernel(3,"Dgemm");
				local_PMD->subkernel_list[current_ctr]->iloc1 = mi;
				local_PMD->subkernel_list[current_ctr]->iloc2 = ni;
				local_PMD->subkernel_list[current_ctr]->iloc3 = ki;
				local_PMD->subkernel_list[current_ctr]->TileList[0] = local_PMD->decom[0]->getTile(mi,ki);
				local_PMD->subkernel_list[current_ctr]->TileList[1] = local_PMD->decom[1]->getTile(ki,ni);
				local_PMD->subkernel_list[current_ctr]->TileList[2] = local_PMD->decom[2]->getTile(mi,ni);
				local_PMD->subkernel_list[current_ctr]->TileList[0]->set_WRP(RONLY);
				local_PMD->subkernel_list[current_ctr]->TileList[1]->set_WRP(RONLY);
				if (!strcmp(OUTPUT_ALGO_MODE, "ALGO_WR"))
					local_PMD->subkernel_list[current_ctr]->TileList[2]->set_WRP(WR);
				else if (!strcmp(OUTPUT_ALGO_MODE, "ALGO_WR_LAZY"))
					local_PMD->subkernel_list[current_ctr]->TileList[2]->set_WRP(WR_LAZY);
				else if (!strcmp(OUTPUT_ALGO_MODE, "ALGO_WREDUCE"))
					local_PMD->subkernel_list[current_ctr]->TileList[2]->set_WRP(W_REDUCE);
				else error("CreateSubkernelsDgemm: Unknown OUTPUT_ALGO_MODE =  %s\n", OUTPUT_ALGO_MODE);
				local_PMD->subkernel_list[current_ctr]->TileList[2]->W_pending = local_PMD->decom[0]->GridSz2;
				local_PMD->subkernel_list[current_ctr]->TileList[2]->reduce_mult = initial_dgemm->beta; 
				local_PMD->subkernel_list[current_ctr]->operation_params = 
					(void*) malloc(sizeof( gemm_backend_in<double>));
				gemm_backend_in<double>*  ptr_ker_translate = 
					(gemm_backend_in<double>*) local_PMD->subkernel_list[current_ctr]->operation_params;
				ptr_ker_translate->TransA = initial_dgemm->TransA;
				ptr_ker_translate->TransB = initial_dgemm->TransB;
				ptr_ker_translate->M = local_PMD->subkernel_list[current_ctr]->TileList[2]->dim1;
				ptr_ker_translate->N = local_PMD->subkernel_list[current_ctr]->TileList[2]->dim2;
				if (ptr_ker_translate->TransA == 'N') ptr_ker_translate->K = 
					local_PMD->subkernel_list[current_ctr]->TileList[0]->dim2;
				else if (ptr_ker_translate->TransA == 'T') ptr_ker_translate->K = 
					local_PMD->subkernel_list[current_ctr]->TileList[0]->dim1;
				else error("CreateSubkernelsDgemm: Unknown transpose type\n");
				ptr_ker_translate->A = NULL;
				ptr_ker_translate->B = NULL;
				ptr_ker_translate->C = NULL;
				ptr_ker_translate->alpha = initial_dgemm->alpha;
				ptr_ker_translate->beta = initial_dgemm->beta;
			}
		}
	}

#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void UpdateSubkernelsDgemm(PMD_p local_PMD){
	gemm_backend_in<double>* initial_dgemm = (gemm_backend_in<double>*) local_PMD->problem_wrap;
	int current_ctr = 0;
	for (int mi = 0; mi < local_PMD->decom[0]->GridSz1; mi++){
		for (int ni = 0; ni < local_PMD->decom[1]->GridSz2; ni++){
			for (int ki = 0; ki < local_PMD->decom[0]->GridSz2; ki++){
	      		current_ctr = mi*local_PMD->decom[1]->GridSz2*local_PMD->decom[0]->GridSz2 
							+ ni*local_PMD->decom[0]->GridSz2 + ki;
				local_PMD->subkernel_list[current_ctr]->TileList[2]->W_pending = local_PMD->decom[0]->GridSz2;
				local_PMD->subkernel_list[current_ctr]->TileList[2]->reduce_mult = initial_dgemm->beta;
				//local_PMD->subkernel_list[current_ctr]->TileList[2]->W_complete->reset();
				gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>*) 
					local_PMD->subkernel_list[current_ctr]->operation_params;
				//ptr_ker_translate->TransA = initial_dgemm->TransA;
				//ptr_ker_translate->TransB = initial_dgemm->TransB;
				//if (ptr_ker_translate->TransA == 'N') ptr_ker_translate->K = 
				//	local_PMD->subkernel_list[current_ctr]->TileList[0]->dim2;
				//else if (ptr_ker_translate->TransA == 'T') ptr_ker_translate->K = 
				//	local_PMD->subkernel_list[current_ctr]->TileList[0]->dim1;
				//else error("CreateSubkernelsDgemm: Unknown transpose type\n");
				ptr_ker_translate->A = NULL;
				ptr_ker_translate->B = NULL;
				ptr_ker_translate->C = NULL;
				ptr_ker_translate->alpha = initial_dgemm->alpha;
				ptr_ker_translate->beta = initial_dgemm->beta;

				int dev_id_idx = idxize(local_PMD->subkernel_list[current_ctr]->run_dev_id);
				local_PMD->subkernel_list[current_ctr]->TileList[0]->try_set_loc_idx(dev_id_idx, 1);
				local_PMD->subkernel_list[current_ctr]->TileList[1]->try_set_loc_idx(dev_id_idx, 1);
				local_PMD->subkernel_list[current_ctr]->TileList[2]->try_set_loc_idx(dev_id_idx, 1);
				if (local_PMD->subkernel_list[current_ctr]->TileList[2]->WRP != WR && 
					!local_PMD->subkernel_list[current_ctr]->TileList[2]->loc_map[dev_id_idx]) 
					local_PMD->subkernel_list[current_ctr]->TileList[2]->set_WRP(WR);
			}
		}
	}
}

void DgemmBindDevice(PMD_p local_PMD, Subkernel* ker, int dev_id){
	gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>* ) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	CoCoPeLiaSelectDevice(dev_id);
#ifdef DEBUG
	fprintf(stderr, "|-----> DgemmBindDevice - Subkernel(dev=%d, id = %d)\n", dev_id, ker->id);
#endif
	int dev_id_idx = idxize(dev_id);
	ptr_ker_translate->ldA = ker->TileList[0]->get_chunk_size(dev_id_idx);
	ptr_ker_translate->ldB = ker->TileList[1]->get_chunk_size(dev_id_idx);
	ptr_ker_translate->ldC = ker->TileList[2]->get_chunk_size(dev_id_idx);
	ker->TileList[0]->try_set_loc_idx(dev_id_idx, 1);
	ker->TileList[1]->try_set_loc_idx(dev_id_idx, 1);
	ker->TileList[2]->try_set_loc_idx(dev_id_idx, 1);
	if (ker->TileList[2]->WRP != WR && !ker->TileList[2]->loc_map[dev_id_idx]) 
		ker->TileList[2]->set_WRP(WR);
	ker->TileList[2]->W_master = dev_id;
	if(ker->TileList[2]->W_pending == local_PMD->decom[0]->GridSz2){
		ker->TileList[2]->W_complete = new Event(dev_id);
		ker->TileList[2]->W_reduce = new Event(ker->TileList[2]->get_initial_location());
	}
	//if(local_PMD->autotuner) if(local_PMD->autotuner->unit_modeler_list[dev_id_idx])
	//	ker->run_op_estimate(local_PMD->autotuner->unit_modeler_list[dev_id_idx]); 
#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
}

void DgemmPrepareLaunch(Subkernel* ker){
	gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>* ) ker->operation_params;
	if(!(ker->TileList[2]->W_master_backend_ctr == -42)) 
	// Means its not the first subkernel using the WR tile
		ptr_ker_translate->beta = 1.0;
	else if(WR_LAZY == ker->TileList[2]->WRP || W_REDUCE == ker->TileList[2]->WRP) ptr_ker_translate->beta = 0;
}

void DgemmUpdatePointers(Subkernel* ker){
	gemm_backend_in<double>*  ptr_ker_translate = (gemm_backend_in<double>* ) ker->operation_params;
	short dev_id_idx = idxize(ker->run_dev_id);
	ptr_ker_translate->A = &ker->TileList[0]->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->B = &ker->TileList[1]->StoreBlock[dev_id_idx]->Adrs;
	ptr_ker_translate->C = &ker->TileList[2]->StoreBlock[dev_id_idx]->Adrs;
}

#ifdef SUBKERNELS_FIRE_WHEN_READY
void* subkernel_manager_wrap(void* dummy){
	int sk_ctr = 0, remaining_sk = PMD_cache[PMD_cache_entries-1]->sk_num; 
	short sk_fired[PMD_cache[PMD_cache_entries-1]->sk_num] = {0};
	while (remaining_sk){
		if(!sk_fired[sk_ctr]){
			Subkernel * curr = PMD_cache[PMD_cache_entries-1]->subkernel_list[sk_ctr];
			if(curr->launched) sk_fired[sk_ctr] = curr->check_ready();
			if(sk_fired[sk_ctr]){
				DgemmUpdatePointers(curr);
				DgemmPrepareLaunch(curr);
				curr->run_ready_operation();
				remaining_sk--;
				//fprintf(stderr, "Fired SK %d\n",sk_ctr);
			}
		}
		if (sk_ctr < PMD_cache[PMD_cache_entries-1]->sk_num - 1) sk_ctr++;
		else{
			sk_ctr = 0; 
			usleep(100); // TODO: This exists solely for nsight profiling reasons
			//fprintf(stderr, "sk_fired = %s, remaining_sk = %d\n",printlist(sk_fired, PMD_cache[PMD_cache_entries-1]->sk_num), remaining_sk);
		}
		//fprintf(stderr, "loop %d ",sk_ctr);
	}
	return NULL; 
}
#endif

/// A dgemm wrapper including auto-tuning of T and cache_size, as well as device management
ATC_p PARALiADgemm(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA,
		double* B, long int ldB, double beta, double* C, long int ldC)
{
	short lvl = 1;
#ifdef DEBUG
	fprintf(stderr, "|-----> PARALiADgemm(%c,%c,%zu,%zu,%zu,%lf,A=%p(%d),%zu,B=%p(%d),%zu,%lf,C=%p(%d),%zu)\n",
		TransA, TransB, M, N, K, alpha, A, CoCoGetPtrLoc(A), ldA,
		B, CoCoGetPtrLoc(B), ldB, beta, C, CoCoGetPtrLoc(C), ldC);
#endif
#ifdef STEST
	gemm_entry_ts = csecond();
#endif
#ifdef TEST
	fprintf(stderr, "|-----> PARALiADgemm\n");
	double cpu_timer = csecond();
#endif
  	signal(SIGSEGV, handler);   // install segfault handler

	int prev_dev_id = CoCoPeLiaGetDevice();

	int reuse_problem_flag = 0;
	PMD_p local_PMD = NULL; 
	gemm_backend_in<double>* initial_dgemm;

	for(int cache_entries = 0; cache_entries < PMD_cache_entries; cache_entries++)
	if(PMD_cache[cache_entries] && !strcmp(PMD_cache[cache_entries]->problem_name, "Dgemm")){
			initial_dgemm = (gemm_backend_in<double>*) PMD_cache[cache_entries]->problem_wrap;
			reuse_problem_flag = 1; 
			if(initial_dgemm->TransA != TransA)
			reuse_problem_flag = 0;
			if(initial_dgemm->TransB != TransB)
				reuse_problem_flag = 0;
			if(initial_dgemm->M != M)
				reuse_problem_flag = 0;
			if(initial_dgemm->N != N)
				reuse_problem_flag = 0;
			if(initial_dgemm->K != K)
				reuse_problem_flag = 0;
			if(initial_dgemm->A!= NULL && CoCoGetPtrLoc(*initial_dgemm->A) != CoCoGetPtrLoc(A))
				reuse_problem_flag = 0;

			if(initial_dgemm->B!= NULL && CoCoGetPtrLoc(*initial_dgemm->B) != CoCoGetPtrLoc(B))
				reuse_problem_flag = 0;

			if(initial_dgemm->C!= NULL && CoCoGetPtrLoc(*initial_dgemm->C) != CoCoGetPtrLoc(C))
				reuse_problem_flag = 0;
			if(reuse_problem_flag){
				local_PMD = PMD_cache[cache_entries]; 
				break; 
			}
	}
	if (!local_PMD){
#ifdef DEBUG
			fprintf(stderr, "No previous problem metadata cache entry found, starting...\n");
#endif
			if (PMD_cache_entries == PROBLEM_MD_CACHE){
				warning("PARALiADgemm - problem cache full, removing first entry\n");
				delete PMD_cache[0];
				local_PMD = PMD_cache[0] = new ProblemMetadata();
			}
			else local_PMD = PMD_cache[PMD_cache_entries++] = new ProblemMetadata();
			local_PMD->problem_wrap = malloc(sizeof(gemm_backend_in<double>));
	}
	initial_dgemm = (gemm_backend_in<double>*) local_PMD->problem_wrap;
	initial_dgemm->TransA = TransA;
	initial_dgemm->TransB = TransB;
	initial_dgemm->M = M;
	initial_dgemm->N = N;
	initial_dgemm->K = K;
	initial_dgemm->A = (void**) &A;
	initial_dgemm->B = (void**) &B;
	initial_dgemm->C = (void**) &C;
	initial_dgemm->alpha = alpha;
	initial_dgemm->beta = beta;
	initial_dgemm->ldA = ldA;
	initial_dgemm->ldB = ldB;
	initial_dgemm->ldC = ldC;
	initial_dgemm->dev_id = -1;

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("PARALiADgemm: pthread_attr_init failed s=%d\n", s);
	void* res;
	double autotune_timer = 0;
	long int T; 
	int remaining_Subkernels = 0;
	int remaining_Subkernels_dev[LOC_NUM] = {0};

	if(!reuse_problem_flag){
		local_PMD->problem_name = "Dgemm";
		local_PMD->decom_num = 3;
		local_PMD->decom[0] = new Decom2D( (void*) A, M, K, ldA, TransA, DOUBLE);
		local_PMD->decom[1] = new Decom2D( (void*) B, K, N, ldB, TransB, DOUBLE);
		local_PMD->decom[2] = new Decom2D( (void*) C, M, N, ldC, 'N', DOUBLE);

		pthread_t asset_thread_id[3];
		local_PMD->decom[0]->prepareAsync(&asset_thread_id[0], attr);
		local_PMD->decom[1]->prepareAsync(&asset_thread_id[1], attr);
		local_PMD->decom[2]->prepareAsync(&asset_thread_id[2], attr);

		local_PMD->autotuner = new ATC();
		if (predef_controller_dgemm && local_PMD->autotuner->diff_intialized_params_ATC(predef_controller_dgemm))
			local_PMD->autotuner->mimic_ATC(predef_controller_dgemm);
		autotune_timer = local_PMD->autotuner->autotune_problem("Dgemm", initial_dgemm);

		for(int d=0; d < LOC_NUM; d++) CoCoEnableLinks(deidxize(d), LOC_NUM);

		for(int i=0; i<3;i++){
			s = pthread_join(asset_thread_id[i], &res);
			if (s != 0) error("PARALiADgemm: pthread_join failed with exit value %d", s);
			//free(res);      /* Free memory allocated by thread */
		}
#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		fprintf(stderr, "Preparing decomposers (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
		cpu_timer = csecond();
#endif
		for (int i = 0; i < LOC_NUM; i++) current_SAB[i] = NULL;
		ManageCachesDgemm(local_PMD);
		T = local_PMD->autotuner->T;
		local_PMD->decom[0]->InitTileMap(T, T, local_PMD->SAB);
		local_PMD->decom[1]->InitTileMap(T, T, local_PMD->SAB);
		local_PMD->decom[2]->InitTileMap(T, T, local_PMD->SAB);
#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		fprintf(stderr, "Decomposing data to tiles -> t_tile = %lf ms\n", cpu_timer*1000);
		cpu_timer = csecond();
#endif
		CreateSubkernelsDgemm(local_PMD);
		remaining_Subkernels = local_PMD->sk_num;
		for(int d=0; d < local_PMD->autotuner->active_unit_num; d++){
			if(local_PMD->autotuner->Subkernels_per_unit_num[d] == 0 )
				error("CoCoPeLiaDgemm: Leftover local_PMD->autotuner->Subkernels_per_unit_num[%d] == 0", d);
			int dev_id = local_PMD->autotuner->active_unit_id_list[d];
			remaining_Subkernels_dev[d] =  local_PMD->sk_dev_num[d] = 
			local_PMD->autotuner->Subkernels_per_unit_num[d];
			local_PMD->subkernel_dev_list[d] = (Subkernel**) 
				malloc(local_PMD->autotuner->Subkernels_per_unit_num[d]*sizeof(Subkernel*));
			for(int sk_ctr = 0; sk_ctr < remaining_Subkernels_dev[d]; sk_ctr++){
				local_PMD->subkernel_dev_list[d][sk_ctr] = local_PMD->subkernel_list
					[local_PMD->autotuner->Subkernels_per_unit_list[d][sk_ctr]];
				DgemmBindDevice(local_PMD, local_PMD->subkernel_dev_list[d][sk_ctr], dev_id);
			}
		}
	}
	else{
		int buffer_freed = 0; 
		for (int i = 0; i < LOC_NUM; i++){
			current_SAB[i] = local_PMD->SAB[i];
			if(!current_SAB[i]) buffer_freed = 1; 
		}
		if(buffer_freed) ManageCachesDgemm(local_PMD);
		T = local_PMD->autotuner->T;
		local_PMD->decom[0]->Reset((void*) A, T, T, ldA, local_PMD->SAB);
		local_PMD->decom[1]->Reset((void*) B, T, T, ldB, local_PMD->SAB);
		local_PMD->decom[2]->Reset((void*) C, T, T, ldC, local_PMD->SAB);
#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		fprintf(stderr, "Re-assigning cache blocks to tiles -> t_tile = %lf ms\n", cpu_timer*1000);
		cpu_timer = csecond();
#endif
		UpdateSubkernelsDgemm(local_PMD);
		remaining_Subkernels = local_PMD->sk_num;
		for(int d=0; d < local_PMD->autotuner->active_unit_num; d++)
			remaining_Subkernels_dev[d] =  local_PMD->sk_dev_num[d];
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Updated subkernels for devices: t_update = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
	for(int d=0; d < local_PMD->autotuner->active_unit_num; d++)
		current_SAB[idxize(local_PMD->autotuner->active_unit_id_list[d])]->allocate(true);
	if(!strcmp(OUTPUT_ALGO_MODE,"ALGO_WREDUCE")){
		reduce_loc = CoCoGetPtrLoc(C); 
		current_SAB[idxize(reduce_loc)]->allocate(true);
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Memory management: t_mem = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoPeLiaInitResources(local_PMD->autotuner->active_unit_id_list, local_PMD->autotuner->active_unit_num);
	//CoCoPeLiaInitWS(local_PMD->autotuner->active_unit_id_list, local_PMD->autotuner->active_unit_num);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Queue/Handle init: t_resource = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
	double run_timer = cpu_timer; 
#endif
#ifdef SUBKERNELS_FIRE_WHEN_READY
	pthread_t manager_thread_id;
	s = pthread_create(&manager_thread_id, &attr,
                                  &subkernel_manager_wrap, NULL);
	while (remaining_Subkernels){
		for(int d_ctr=0; d_ctr < local_PMD->autotuner->active_unit_num; d_ctr++){
			//int d = d_ctr*2 % (local_PMD->autotuner->active_unit_num) + d_ctr*2 / (local_PMD->autotuner->active_unit_num); 
			//printf("d_ctr(%d) = d(%d)\n", d_ctr, d); 
			int d = d_ctr;
			if (remaining_Subkernels_dev[d]){
				int dev_id = local_PMD->autotuner->active_unit_id_list[d];
				Subkernel * curr = NULL;
				if (reuse_problem_flag){
					for (int sk_ctr = 0; sk_ctr < local_PMD->sk_dev_num[d]; sk_ctr++)
						if(local_PMD->subkernel_dev_list[d][sk_ctr]->launch_order ==  
						(local_PMD->sk_dev_num[d] - remaining_Subkernels_dev[d]) + 1)
							curr = local_PMD->subkernel_dev_list[d][sk_ctr];
				}
				else curr = SubkernelSelect(dev_id, local_PMD->subkernel_dev_list[d], 
					local_PMD->sk_dev_num[d]);
				if (!curr){
					warning("PARALiADgemm - dev(%d): Got curr = NULL, repeating search\n", dev_id);
					continue;
				}
				remaining_Subkernels_dev[d]--;
				remaining_Subkernels--;
				curr->request_data();
				curr->launch_order =  local_PMD->sk_dev_num[d] - remaining_Subkernels_dev[d]; 
				curr->launched = 1; 

			}
		}
	}
	s = pthread_join(manager_thread_id, &res);
	if (s != 0) error("PARALiADgemm: manager_thread_id failed with exit value %d", s);
#else
	//Subkernel* prev[LOC_NUM] = {NULL};
	while (remaining_Subkernels){
		for(int d=0; d < local_PMD->autotuner->active_unit_num; d++) if (remaining_Subkernels_dev[d]){
			int dev_id = local_PMD->autotuner->active_unit_id_list[d];
			//if(prev[d]) prev[d]->sync_request_data();
			Subkernel * curr = NULL;
			if (reuse_problem_flag){
				for (int sk_ctr = 0; sk_ctr < local_PMD->sk_dev_num[d]; sk_ctr++)
					if(local_PMD->subkernel_dev_list[d][sk_ctr]->launch_order ==  
					( local_PMD->sk_dev_num[d] - remaining_Subkernels_dev[d]) + 1)
						curr = local_PMD->subkernel_dev_list[d][sk_ctr];
			}
			else curr = SubkernelSelect(dev_id, local_PMD->subkernel_dev_list[d], 
				local_PMD->sk_dev_num[d]);
			if (!curr){
				warning("PARALiADgemm - dev(%d): Got curr = NULL, repeating search\n", dev_id);
				continue;
			}
			remaining_Subkernels_dev[d]--;
			remaining_Subkernels--;
			DgemmPrepareLaunch(curr);
			curr->request_data();
			DgemmUpdatePointers(curr);
			curr->run_operation();
			curr->launch_order =  local_PMD->sk_dev_num[d] - remaining_Subkernels_dev[d]; 
			curr->launched = 1; 
			//prev[d] = curr; 
		}
		//usleep(3000);
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Subkernels launched: t_sk_fire = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
	local_PMD->decom[2]->WBTileMap();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Writebacks launched -> t_wb_fire = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
#endif
#ifdef SUBKERNELS_FIRE_WHEN_READY
#ifndef ENABLE_SEND_RECV_OVERLAP
	sync_recv_queues();
	local_PMD->decom[2]->WBTileMap();
#endif
#endif
	local_PMD->decom[2]->SyncTileMap();
	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - run_timer;
	fprintf(stderr, "Synced result -> t_exec_full = %lf ms\n", cpu_timer*1000);
	fprintf(stderr, "t_predicted for T=%zu was %.2lf ms : %lf percentile error\n", T, local_PMD->autotuner->pred_t*1000,
	(local_PMD->autotuner->pred_t==0)? 0.0: (local_PMD->autotuner->pred_t - cpu_timer )/local_PMD->autotuner->pred_t*100);
	cpu_timer = csecond();
#endif
#ifdef PDEBUG
	fprintf(stderr, "PARALiADgemm(): completed for PMD_cache_entries = %d\n", PMD_cache_entries);
#endif
#ifdef STEST
	STEST_print_SK(thread_dev_data, gemm_entry_ts, local_PMD->autotuner->active_unit_num);
#endif

#ifdef TTEST
	HopMemcpyPrint();
	//n_HopMemcpyPrint();
#endif

#ifdef DDEBUG
  local_PMD->decom[0]->DrawTileMap();
  local_PMD->decom[1]->DrawTileMap();
  local_PMD->decom[2]->DrawTileMap();
#endif

#ifdef CDEBUG
	for(int i=0; i<LOC_NUM;i++) local_PMD->SAB[i]->draw_buffer(true,true,true);
#endif

#ifdef BUFFER_REUSE_ENABLE
	for(int i=0; i<LOC_NUM;i++) local_PMD->SAB[i]->reset(false,true);
#else
	for(int i = 0 ; i < LOC_NUM; i++){
		delete local_PMD->SAB[i];
		local_PMD->SAB[i] = current_SAB[i] = NULL;
	}
#endif

#ifdef QUEUE_REUSE_ENABLE
	CoCoPeLiaCleanResources();
#else 
	CoCoPeLiaFreeResources();
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

#ifdef METADATA_REUSE_PROBLEMS
	for(int i=0; i<local_PMD->sk_num; i++) local_PMD->subkernel_list[i]->reset();
#else
	for(int i=0; i<local_PMD->sk_num; i++) delete local_PMD->subkernel_list[i];
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Freed Subkernels -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
	local_PMD->decom[0]->DestroyTileMap();
	local_PMD->decom[1]->DestroyTileMap();
	local_PMD->decom[2]->DestroyTileMap();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Destroyed Tilemaps -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	 CoCoPeLiaSelectDevice(prev_dev_id);

    local_PMD->decom[0]->resetProperties();
    local_PMD->decom[1]->resetProperties();
    local_PMD->decom[2]->resetProperties();
	delete local_PMD->decom[0];
	delete local_PMD->decom[1];
	delete local_PMD->decom[2];

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	fprintf(stderr, "Unregistering decomposers -> t_unpin = %lf ms\n", cpu_timer*1000);
#endif
#endif

#ifdef DEBUG
	fprintf(stderr, "<-----|\n");
#endif
#ifdef TEST
	fprintf(stderr, "<-----|\n");
#endif

	predef_controller_dgemm = NULL;
	// Better not return our global to the user, he can accidentally do stuff to it.
	ATC_p result = new ATC();
	result->mimic_ATC(local_PMD->autotuner);
	return result;
}

/// A modification of PARALiADgemm but with given parameters (mainly for performance/debug purposes)
ATC_p PARALiADgemmControled(char TransA,  char TransB, long int M, long int N, long int K, double alpha, double* A, long int ldA,
		double* B, long int ldB, double beta, double* C, long int ldC, ATC_p predef_controller){
	if (predef_controller == NULL){
		warning("Calling PARALiADgemmControled with empty controller -> falling back to full autotune version \'PARALiADgemm\'\n");
		return PARALiADgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
	}
	predef_controller_dgemm = predef_controller;
	return PARALiADgemm(TransA, TransB,  M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}
