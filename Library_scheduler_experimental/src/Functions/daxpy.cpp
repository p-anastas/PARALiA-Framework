///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The axpy CoCopeLia implementation using the new mission-agent-asset C++ classes.
///

#include "backend_wrappers.hpp"
#include "CoCoPeLiaModel.hpp"
#include "CoCoPeLia.hpp"
#include "unihelpers.hpp"
#include "Asset.hpp"
#include "Subkernel.hpp"
#include "DataCaching.hpp"

#include <pthread.h>

axpy_backend_in_p initial_daxpy = NULL;

CoCoModel_p glob_model_daxpy;
struct CoControl predef_vals_daxpy;
CoControl_p used_vals_daxpy = NULL;
int NGridSz_daxpy = 0;

void CoCoDaxpyUpdateDevice(Subkernel* ker, short dev_id){
	axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) ker->operation_params;
	ker->run_dev_id = ptr_ker_translate->dev_id = dev_id;
	short dev_id_idx = (dev_id == -1) ? LOC_NUM - 1: dev_id;
	ptr_ker_translate->x = &((Tile1D<VALUE_TYPE>*) ker->TileList[0])->adrs[dev_id_idx];
	ptr_ker_translate->y = &((Tile1D<VALUE_TYPE>*) ker->TileList[1])->adrs[dev_id_idx];
	ptr_ker_translate->incx = ((Tile1D<VALUE_TYPE>*) ker->TileList[0])->inc[dev_id_idx];
	ptr_ker_translate->incy = ((Tile1D<VALUE_TYPE>*) ker->TileList[1])->inc[dev_id_idx];
}

Subkernel** CoCoAsignTilesToSubkernelsDaxpy(Asset1D<VALUE_TYPE>* x_asset, Asset1D<VALUE_TYPE>* y_asset,
	int T, int* kernelNum){

	short lvl = 2;

	NGridSz_daxpy = x_asset->GridSz;
	*kernelNum = NGridSz_daxpy;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoAsignTilesToSubkernelsDaxpy(x_asset,y_asset,%d,%d)\n", T, *kernelNum);
	lprintf(lvl,"NGridSz_daxpy = %d\n", NGridSz_daxpy);
	lprintf(lvl,"Nlast = %d\n",
	x_asset->Tile_map[NGridSz_daxpy-1]->dim);
#endif

Subkernel** kernels = (Subkernel**) malloc(*kernelNum*sizeof(Subkernel*));
int current_ctr = 0;
		for (int ni = 0; ni < NGridSz_daxpy; ni++){
      current_ctr = ni;
			kernels[current_ctr] = new Subkernel(2,"axpy");
			kernels[current_ctr]->iloc1 = ni;
			kernels[current_ctr]->TileDimlist[0] = kernels[current_ctr]->TileDimlist[1] = 1;
			kernels[current_ctr]->TileList[0] = x_asset->getTile(ni);
			kernels[current_ctr]->TileList[1] = y_asset->getTile(ni);
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[0])->R_flag = 1;
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[1])->R_flag = 1;
			((Tile1D<VALUE_TYPE>*)kernels[current_ctr]->TileList[1])->W_flag = 1;
			kernels[current_ctr]->operation_params = (void*) malloc(sizeof(struct axpy_backend_in));
			axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) kernels[current_ctr]->operation_params;
			ptr_ker_translate->N = ((Tile1D<VALUE_TYPE>*) kernels[current_ctr]->TileList[0])->dim;
			ptr_ker_translate->x = NULL;
			ptr_ker_translate->y = NULL;
			ptr_ker_translate->alpha = initial_daxpy->alpha;
			ptr_ker_translate->incx = initial_daxpy->incy;
			// No interal dims for axpy to reduce
			kernels[current_ctr]->WR_first = kernels[current_ctr]->WR_last = 1;
			kernels[current_ctr]->WR_reduce = 0;
		}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}

void* CoCopeLiaDaxpyAgentVoid(void* kernel_pthread_wrapped){
	short lvl = 2;

	kernel_pthread_wrap_p axpy_subkernel_data = (kernel_pthread_wrap_p)kernel_pthread_wrapped;
	short dev_id = axpy_subkernel_data->dev_id;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDaxpyAgentVoid(axpy_subkernel_data: dev_id = %d, SubkernelNumDev = %d)\n",
		dev_id, axpy_subkernel_data->SubkernelNumDev);
#endif
#ifdef TEST
		double cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(dev_id);

	for (int keri = 0; keri < axpy_subkernel_data->SubkernelNumDev; keri++){
		axpy_subkernel_data->SubkernelListDev[keri]->init_events();
		CoCoDaxpyUpdateDevice(axpy_subkernel_data->SubkernelListDev[keri], dev_id);
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Update Subkernels -Init Events(%d): t_update = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	/// Only works assuming the last subkernel writes back
	Event* tmp_writeback;
	for (int keri = axpy_subkernel_data->SubkernelNumDev -1 ; keri >= 0 ; keri--){
		if (axpy_subkernel_data->SubkernelListDev[keri]->WR_last)
			tmp_writeback = axpy_subkernel_data->SubkernelListDev[keri]->writeback_complete;
		else axpy_subkernel_data->SubkernelListDev[keri]->writeback_complete = tmp_writeback;
		}

#ifdef TEST
	cpu_timer = csecond();
#endif
  CoCoPeLiaRequestBuffer(axpy_subkernel_data, used_vals_daxpy->cache_limit, used_vals_daxpy->T*sizeof(VALUE_TYPE));
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Memory management(%d): t_mem = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif
	CoCoPeLiaInitResources(dev_id);
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Stream/Lib Handle Initialization(%d): t_resource = %lf ms\n", dev_id, cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for (int keri = 0; keri < axpy_subkernel_data->SubkernelNumDev; keri++){
		if (!keri) axpy_subkernel_data->SubkernelListDev[keri]->prev = NULL;
		else axpy_subkernel_data->SubkernelListDev[keri]->prev =
			axpy_subkernel_data->SubkernelListDev[keri-1];
		if(keri==axpy_subkernel_data->SubkernelNumDev - 1)
			axpy_subkernel_data->SubkernelListDev[keri]->next = NULL;
		else axpy_subkernel_data->SubkernelListDev[keri]->next =
			axpy_subkernel_data->SubkernelListDev[keri+1];
		axpy_subkernel_data->SubkernelListDev[keri]->request_data();
		axpy_subkernel_data->SubkernelListDev[keri]->run_operation();
		if (axpy_subkernel_data->SubkernelListDev[keri]->WR_last)
			axpy_subkernel_data->SubkernelListDev[keri]->writeback_data();
	}

	CoCoSyncCheckErr();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernels complete(%d): t_comp = %lf ms\n" , dev_id, cpu_timer*1000);
#endif
	/// Do this after pthread join to enable other devices
	/// to still read cached data after a device's part is over
	//CoCoPeLiaDevCacheInvalidate(axpy_subkernel_data);
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return NULL;
}

/// An axpy wrapper including auto-tuning of T and cache_size, as well as device management
CoControl_p CoCopeLiaDaxpy(size_t N, VALUE_TYPE alpha, VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy)
{
	short lvl = 1;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDaxpy(%zu,%lf,x=%p(%d),%zu,y=%p(%d),%zu)\n",
		N, alpha, x, CoCoGetPtrLoc(x), incx, y, CoCoGetPtrLoc(y), incy);
#endif

#ifdef TEST
	lprintf(lvl-1, "|-----> CoCopeLiaDaxpy\n");
	double cpu_timer = csecond();
#endif

	int prev_dev_id = CoCoPeLiaGetDevice();

	if(!initial_daxpy) initial_daxpy = (axpy_backend_in_p) malloc(sizeof(struct axpy_backend_in));
	initial_daxpy->N = N;
	initial_daxpy->x = (void**) &x;
	initial_daxpy->y = (void**) &y;
	initial_daxpy->alpha = alpha;
	initial_daxpy->incx = incx;
	initial_daxpy->incy = incy;
	initial_daxpy->dev_id = -1;

	Asset1D<VALUE_TYPE>* x_asset, *y_asset;
	/// Prepare Assets in parallel( e.g. initialize asset classes, pin memory with pthreads)
	/// return: x_asset, y_asset initialized and pinned
	{
		x_asset = new Asset1D<VALUE_TYPE>( x, N, incx);
		y_asset = new Asset1D<VALUE_TYPE>( y, N, incy);

		pthread_attr_t attr;
		int s = pthread_attr_init(&attr);
		if (s != 0) error("CoCopeLiaDaxpy: pthread_attr_init failed s=%d\n", s);

		pthread_t asset_thread_id[2];
		x_asset->prepareAsync(&asset_thread_id[0], attr);
		y_asset->prepareAsync(&asset_thread_id[1], attr);

		void* res;
		for(int i=0; i<2;i++){
			s = pthread_join(asset_thread_id[i], &res);
			if (s != 0) error("CoCopeLiaDaxpy: pthread_join failed with exit value %d", s);
			//free(res);      /* Free memory allocated by thread */
		}

#ifdef TEST
		cpu_timer = csecond() - cpu_timer;
		lprintf(lvl, "Preparing assets (parallel with pthreads) -> t_prep = %lf ms\n", cpu_timer*1000);
		cpu_timer = csecond();
#endif
	}

	/// Read predefined values for device selection or use default.
	/// return: num_devices, dev_id initialized, update used_vals_daxpy
	short num_devices = 0, *dev_id = NULL;
	{
		if (predef_vals_daxpy.dev_num > 0){
			num_devices = predef_vals_daxpy.dev_num;
			dev_id = (short*) malloc (num_devices*sizeof(short));
			for (int i =0; i < num_devices; i++) dev_id[i] = predef_vals_daxpy.dev_ids[i];
#ifdef DEBUG
			lprintf(lvl, "Running on %d devices with dev_ids=[ ", num_devices);
			for (int i =0; i < num_devices; i++) fprintf(stderr, "%d ", predef_vals_daxpy.dev_ids[i]);
			fprintf(stderr, "]\n");
#endif
		}
		else if (predef_vals_daxpy.dev_num == 0) error("CoCopeLiaDaxpy: CPU-only version not implemented\n");
		else{
#ifdef ENABLE_CPU_WORKLOAD
			num_devices = LOC_NUM;
#else
			num_devices = DEV_NUM;
#endif
			dev_id = (short*) malloc (num_devices*sizeof(short));
			for (int i = 0; i < num_devices; i++) dev_id[i] = (i != LOC_NUM - 1)? i : -1;
		}

		if(used_vals_daxpy == NULL) {
			used_vals_daxpy = (CoControl_p) malloc(sizeof(struct CoControl));
		}
		used_vals_daxpy->dev_num = num_devices;
		for (int d = 0; d< num_devices; d++) used_vals_daxpy->dev_ids[d] = dev_id[d];
	}

	/// Read predefined values for T or use Tile selection.
	/// return: T size for datum
	size_t T = 256;
	double slowest_problem_t = 0;
	CoCoModel_p model = NULL;
	{
		if(predef_vals_daxpy.T <= 0){
			error("daxpy prediction under construction\n");
			/*
			/// Naive for multiple equivalent devices.
			int slowest_problem_T = std::min((size_t) 512*512, N);
			tunableParams_p pred_p[num_devices];
			for (int d = 0 ; d < num_devices; d++){
				model = CoCoPeLiaModelInit(dev_id[d], "Daxpy", 'X', 'X', 'X',
					N/num_devices, 0, 0,
					(CoCoGetPtrLoc(x) == dev_id[d])? 0 : 1, (CoCoGetPtrLoc(y) == dev_id[d])? 0 : 1,
					-1, (CoCoGetPtrLoc(x) == dev_id[d])? 0 : 1,
					(CoCoGetPtrLoc(y) == dev_id[d])? 0 : 1, -1,
					incx, incy, 0);
#ifdef TEST
				cpu_timer = csecond() - cpu_timer;
				lprintf(lvl, "Model Initialization(dev = %d): t_mod_init = %lf ms\n", dev_id[d], cpu_timer*1000);
				cpu_timer = csecond();
#endif

				pred_p[d] = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_BIDIRECTIONAL);
				if (pred_p[d]->pred_t > slowest_problem_t){
					slowest_problem_t = pred_p[d]->pred_t;
					slowest_problem_T = pred_p[d]->T;
				}
#ifdef TEST
				cpu_timer = csecond() - cpu_timer;
				lprintf(lvl, "Model Selected T=%zu for dev = %d with t_predicted = %lf ms : t_mod_opt = %lf ms\n", pred_p[d]->T, dev_id[d], pred_p[d]->pred_t*1000, cpu_timer*1000);
				cpu_timer = csecond();
#endif
			}

			/// Extra: check if running in multiple GPUs seems to have a point performance-wise.
			/// Currently only comparing single vs multi GPU
			/// Can be extended to complex (e.g. 1 vs 2 vs 3 etc)
			if (predef_vals_daxpy.dev_num < 0 && num_devices > 1) {
				short best_dev_id = 0;
				model = CoCoPeLiaModelInit(0, "axpy", 'X', 'X', 'X',
					N, 0, 0,
					(CoCoGetPtrLoc(x) == 0)? 0 : 1, (CoCoGetPtrLoc(y) == 0)? 0 : 1,
					-1, (CoCoGetPtrLoc(x) == 0)? 0 : 1,
					(CoCoGetPtrLoc(y) == 0)? 0 : 1, -1,
					incx, incy, 0);

				tunableParams_p pred_p_single_dev = CoCoPeLiaModelOptimizeTile(model, COCOPELIA_BIDIRECTIONAL);

#ifdef TEST
			 cpu_timer = csecond() - cpu_timer;
			 lprintf(lvl, "Model Selected T=%zu for single-device execution(%d) with t_predicted = %lf ms : t_mod_opt = %lf ms\n", pred_p_single_dev->T, best_dev_id, pred_p_single_dev->pred_t*1000, cpu_timer*1000);
			 cpu_timer = csecond();
#endif

				/// How much performance improvent justifies adding one more GPU?
				/// Aren't there better metrics for this?
				if (slowest_problem_t > pred_p_single_dev->pred_t){
				 	slowest_problem_T = pred_p_single_dev->T;
				 	warning("Chose to run on only 1 device: Model implies %lf\% better performance\n",
						(slowest_problem_t - pred_p_single_dev->pred_t)/slowest_problem_t*100);
					slowest_problem_t = pred_p_single_dev->pred_t;
					num_devices = 1;
					dev_id[0] = best_dev_id;
			 	}
			}

			T = slowest_problem_T;
#ifdef TEST
			cpu_timer = csecond() - cpu_timer;
			lprintf(lvl, "Model Selected T=%zu with t_predicted = %lf ms : t_mod_opt = %lf ms\n", T, slowest_problem_t*1000, cpu_timer*1000);
			cpu_timer = csecond();
#endif

#ifdef DEBUG
			lprintf(lvl, "Model Selected T=%zu : t_predicted = %lf ms\n", T, slowest_problem_t*1000);
			lprintf(lvl, "====================================\n");
#endif
*/
		}
		else{
			T = predef_vals_daxpy.T;
#ifdef DEBUG
			lprintf(lvl, "====================================\n");
			lprintf(lvl, "Using predefined T=%zu\n", T);
			lprintf(lvl, "====================================\n");
#endif
		}
		if(used_vals_daxpy == NULL) {
			used_vals_daxpy = (CoControl_p) malloc(sizeof(struct CoControl));
		}
		used_vals_daxpy->T = T;
		used_vals_daxpy->cache_limit = predef_vals_daxpy.cache_limit;
	}

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Device/T selection -> t_configure = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	/// TODO: Split each asset to Tiles
	x_asset->InitTileMap(T);
	y_asset->InitTileMap(T);

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Spliting assets to tiles -> t_tile_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	int Subkernel_num;
	Subkernel** Subkernel_list = CoCoAsignTilesToSubkernelsDaxpy(x_asset, y_asset, T,
		&Subkernel_num);
#ifdef DEBUG
	lprintf(lvl, "Subkernel_num = %d {N}GridSz = {%d}, num_devices = %d\n\n",
		Subkernel_num, NGridSz_daxpy, num_devices);
#endif
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel init -> t_subkernel_init = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif
	tunableParams_p dummy;
	if (!strcmp(DISTRIBUTION, "ROUND-ROBIN"))
		CoCoDistributeSubkernelsRoundRobin(used_vals_daxpy, dummy, NGridSz_daxpy, -1, -1);
	else if (!strcmp(DISTRIBUTION, "SPLITD1-NAIVE"))
		CoCoDistributeSubkernelsNaive(used_vals_daxpy, dummy, NGridSz_daxpy, -1, -1);
	else error("CoCopeLiaDaxpy: Unknown Subkernel Distribution %s\n", DISTRIBUTION);

	pthread_attr_t attr;
	int s = pthread_attr_init(&attr);
	if (s != 0) error("CoCopeLiaDaxpy: pthread_attr_init failed s=%d\n", s);
	void* res;
	int used_devices = 0;
	for (int d = 0 ; d < num_devices; d++) if(used_vals_daxpy->Subkernels_per_dev[d] > 0 ) used_devices++;
	pthread_t thread_id[used_devices];
	kernel_pthread_wrap_p thread_dev_data[used_devices];

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Subkernel Split devices -> t_subkernel_split = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int d=0; d<used_devices;d++){

		// Check/Enable peer access between participating GPUs
		CoCoEnableLinks(d, dev_id, num_devices);

		thread_dev_data[d] = (kernel_pthread_wrap_p) malloc(sizeof(struct kernel_pthread_wrap));
		thread_dev_data[d]->dev_id = dev_id[d];

		thread_dev_data[d]->SubkernelListDev = (Subkernel**) malloc(used_vals_daxpy->Subkernels_per_dev[d]* sizeof(Subkernel*));
		for(int skitt = 0; skitt < used_vals_daxpy->Subkernels_per_dev[d]; skitt++)
			thread_dev_data[d]->SubkernelListDev[skitt] = Subkernel_list[used_vals_daxpy->Subkernel_dev_id_list[d*Subkernel_num + skitt]];

		thread_dev_data[d]->SubkernelNumDev = used_vals_daxpy->Subkernels_per_dev[d];

		s = pthread_create(&thread_id[d], &attr,
                                  &CoCopeLiaDaxpyAgentVoid, thread_dev_data[d]);

	}
	for(int d=0; d<used_devices;d++){
		s = pthread_join(thread_id[d], &res);
		if (s != 0) error("CoCopeLiaDaxpy: pthread_join failed with exit value %d", s);
		//free(res);      /* Free memory allocated by thread */
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Fire and gather pthreads for all devices -> t_exec_full = %lf ms\n", cpu_timer*1000);
	if(predef_vals_daxpy.T <= 0){
		lprintf(lvl, "t_predicted for T=%zu was %.2lf ms : %lf \% error\n",
		T, slowest_problem_t*1000,
		(slowest_problem_t==0)? 0: (slowest_problem_t - cpu_timer )/slowest_problem_t*100);
	}
	cpu_timer = csecond();
#endif

#ifdef MULTIDEVICE_REDUCTION_ENABLE
	CoCoReduceSyncThreads();
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Gathered reduce pthreads for all devices -> t_reduce_extra = %lf ms\n",
		cpu_timer*1000);
	cpu_timer = csecond();
#endif
#endif

#ifndef BUFFER_REUSE_ENABLE
	for(int i=0; i<used_devices;i++) CoCopeLiaDevCacheFree(i);
#else
	for(int i=0; i<used_devices;i++) CoCoPeLiaDevCacheInvalidate(thread_dev_data[i]);
#endif

#ifndef BACKEND_RES_REUSE_ENABLE
	for(int i=0; i<used_devices;i++) CoCoPeLiaFreeResources(i);
#endif

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Invalidate caches -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	for(int i=0; i<Subkernel_num; i++) delete Subkernel_list[i];
	//delete [] Subkernel_list;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Freed Subkernels -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	x_asset->DestroyTileMap();
	y_asset->DestroyTileMap();

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Destroyed Tilemaps -> t_invalidate = %lf ms\n", cpu_timer*1000);
	cpu_timer = csecond();
#endif

	CoCoPeLiaSelectDevice(prev_dev_id);

  x_asset->resetProperties();
  y_asset->resetProperties();
	delete x_asset;
	delete y_asset;

#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Unregistering assets -> t_unpin = %lf ms\n", cpu_timer*1000);
#endif

#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef TEST
	lprintf(lvl-1, "<-----|\n");
#endif
	return used_vals_daxpy;
}

/// A modification of CoCopeLiaDaxpy but with given parameters (mainly for performance/debug purposes)
CoControl_p CoCopeLiaDaxpyControled(size_t N, VALUE_TYPE alpha, VALUE_TYPE* x, size_t incx, VALUE_TYPE* y, size_t incy, CoControl_p predef_control_values){
	if (predef_control_values == NULL) return CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
	predef_vals_daxpy.T = predef_control_values->T;
	predef_vals_daxpy.dev_num = predef_control_values->dev_num;
	for(int idx =0; idx < LOC_NUM; idx++)
		predef_vals_daxpy.dev_ids[idx] = predef_control_values->dev_ids[idx];
	predef_vals_daxpy.cache_limit = predef_control_values->cache_limit;
	return CoCopeLiaDaxpy(N, alpha, x, incx, y, incy);
}
