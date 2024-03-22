///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <cstdio>
#include <typeinfo>
#include <float.h>
#include <curand.h>

#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

#ifdef TTEST /// C programmers hate him
int fast_trans_ctr = 0;
long long bytes[10000] = {0};
int inter_hop_locs[10000][5];
double inter_hop_timers[10000][4][3];
int timer_ctr[LOC_NUM][LOC_NUM] = {{0}};
double link_gbytes_s[LOC_NUM][LOC_NUM] = {{0}};
int hop_log_lock = 0; /// This might slow down things, but it is needed. 

void reseTTEST(){
	for(int k = 0; k < fast_trans_ctr; k++){
				bytes[k] = 0;
				for(int l = 0; l < 5; l++){
					inter_hop_locs[k][l] = -42;
					if(l < 4) for(int m = 0; m < 3; m++) inter_hop_timers[k][l][m] = 0;
				} 
	}
	fast_trans_ctr = 0;
	for (int d1 = 0; d1 < LOC_NUM; d1++)
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			timer_ctr[d1][d2] = 0; 
			link_gbytes_s[d1][d2] = 0; 
		}
}
#endif

//#define SPLIT_2D_ROWISE
void FasTCoCoMemcpy2DAsync(link_road_p roadMap, long int rows, long int cols, short elemSize){
	if(roadMap->hop_num - roadMap->starting_hop < 2) error("FasTCoCoMemcpy2DAsync: Cannot copy with less than 2 locations\n");
#ifdef TTEST
	while(__sync_lock_test_and_set(&hop_log_lock, 1));
	if (roadMap->hop_num > 4) error("FasTCoCoMemcpy2DAsync(dest = %d, src = %d) exeeded 3 intermediate hops in TTEST Mode\n",
			roadMap->hop_uid_list[roadMap->starting_hop], roadMap->hop_uid_list[roadMap->hop_num]);
	if (fast_trans_ctr > 10000) error("FasTCoCoMemcpy2DAsync(dest = %d, src = %d) exeeded 10000 transfers in TTEST Mode\n",
			roadMap->hop_uid_list[roadMap->starting_hop], roadMap->hop_uid_list[roadMap->hop_num]);
	if(!fast_trans_ctr) reseTTEST();
	bytes[fast_trans_ctr] = rows*cols*elemSize;
#endif
	int buffer_bw_overlap = 8;
	if (rows/buffer_bw_overlap < 100) buffer_bw_overlap = (rows/100) + 1 ;
	if (cols/buffer_bw_overlap < 100) buffer_bw_overlap = (cols/100) + 1 ;
	Event_p step_events[roadMap->hop_num][buffer_bw_overlap];
	for(int uid_ctr = roadMap->starting_hop; uid_ctr < roadMap->hop_num - 1; uid_ctr++){
#ifdef SPLIT_2D_ROWISE
		long int local_rows = rows/buffer_bw_overlap;
#else
		long int local_cols = cols/buffer_bw_overlap;
#endif
		for(int steps = 0; steps < buffer_bw_overlap; steps++){
			step_events[uid_ctr][steps] = new Event(roadMap->hop_uid_list[uid_ctr+1]);
#ifdef SPLIT_2D_ROWISE
			long buff_offset_dest = steps* elemSize * local_rows,
			buff_offset_src = steps * elemSize * local_rows;
			if(steps == buffer_bw_overlap -1) local_rows+= rows%buffer_bw_overlap;
#else
			long buff_offset_dest = steps* elemSize * local_cols * roadMap->hop_ldim_list[uid_ctr + 1],
			buff_offset_src = steps * elemSize * local_cols * roadMap->hop_ldim_list[uid_ctr];
			if(steps == buffer_bw_overlap -1) local_cols+= cols%buffer_bw_overlap;
#endif
			if(uid_ctr > 0) roadMap->hop_cqueue_list[uid_ctr]->wait_for_event(step_events[uid_ctr-1][steps]);
#ifdef TTEST
			if(!steps){
				inter_hop_locs[fast_trans_ctr][uid_ctr - roadMap->starting_hop] = roadMap->hop_uid_list[uid_ctr];
				CoCoSetTimerAsync(&(inter_hop_timers[fast_trans_ctr][uid_ctr - roadMap->starting_hop][0]));
				roadMap->hop_cqueue_list[uid_ctr]->add_host_func((void*)&CoCoSetTimerAsync, 
					(void*) &(inter_hop_timers[fast_trans_ctr][uid_ctr - roadMap->starting_hop][1]));
			}
#endif
			CoCoMemcpy2DAsync_noTTs(roadMap->hop_buf_list[uid_ctr + 1] + buff_offset_dest, roadMap->hop_ldim_list[uid_ctr + 1],
										roadMap->hop_buf_list[uid_ctr] + buff_offset_src, roadMap->hop_ldim_list[uid_ctr],
#ifdef SPLIT_2D_ROWISE
										local_rows, cols, elemSize,
#else
										rows, local_cols, elemSize,
#endif
										roadMap->hop_uid_list[uid_ctr + 1], roadMap->hop_uid_list[uid_ctr], roadMap->hop_cqueue_list[uid_ctr]);
			step_events[uid_ctr][steps]->record_to_queue(roadMap->hop_cqueue_list[uid_ctr]);
		}
		roadMap->hop_event_list[uid_ctr]->record_to_queue(roadMap->hop_cqueue_list[uid_ctr]);
#ifdef TTEST
	roadMap->hop_cqueue_list[uid_ctr]->add_host_func((void*)&CoCoSetTimerAsync, 
		(void*) &(inter_hop_timers[fast_trans_ctr][uid_ctr - roadMap->starting_hop][2]));
#endif
	}
#ifdef TTEST
	inter_hop_locs[fast_trans_ctr][roadMap->hop_num - 1] = roadMap->hop_uid_list[roadMap->hop_num-1];
	fast_trans_ctr++;
	__sync_lock_release(&hop_log_lock);
#endif	
}

#ifdef TTEST
void HopMemcpyPrint(){
	lprintf(0,"\n Hop Tranfers Full:\n");
	FILE* fp = fopen("temp_hop_trans.log", "w+");
	for(int k = 0; k < fast_trans_ctr; k++){
		int src = inter_hop_locs[k][0], dest = inter_hop_locs[k][1], iloc = 1;
		for(int l = 2; l < 5; l++) if(inter_hop_locs[k][l]!= -42){
			dest = inter_hop_locs[k][l];
			iloc = l; 
		}
		timer_ctr[idxize(dest)][idxize(src)]++;
		double time = (inter_hop_timers[k][iloc-1][2] - inter_hop_timers[k][0][1]), pipe_time = (inter_hop_timers[k][iloc-1][2] - inter_hop_timers[k][0][0]);
		link_gbytes_s[idxize(dest)][idxize(src)]+=Gval_per_s(bytes[k], time);
		lprintf(0, "Hop Trasfer %d->%d -> road: %s total_t = %lf ms ( %.3lf Gb/s ), pipelined_t = %lf ms ( %.3lf Gb/s )\n", 
			inter_hop_locs[k][0], inter_hop_locs[k][iloc], printlist(inter_hop_locs[k], iloc+1),
		1000*time, Gval_per_s(bytes[k], time), 1000*pipe_time, Gval_per_s(bytes[k], pipe_time));
		fprintf(fp, "%d,%d,%s,%ld,%lf,%lf,%lf\n", inter_hop_locs[k][0], inter_hop_locs[k][iloc], printlist(inter_hop_locs[k], iloc+1), bytes[k], 
			inter_hop_timers[k][0][0], inter_hop_timers[k][0][1], inter_hop_timers[k][iloc-1][2]);
		/*for (int inter_transfers = 0; inter_transfers < iloc ; inter_transfers++){
			double time = (inter_hop_timers[k][inter_transfers][2] - inter_hop_timers[k][inter_transfers][1]), 
			pipe_time = (inter_hop_timers[k][inter_transfers][2] - inter_hop_timers[k][inter_transfers][0]);
			lprintf(1, "link trasfer %d->%d : total_t = %lf ms ( %.3lf Gb/s ), pipelined_t = %lf ms ( %.3lf Gb/s )\n", 
				inter_hop_locs[k][inter_transfers], inter_hop_locs[k][inter_transfers+1], 1000*time, Gval_per_s(bytes[k], time), 1000*pipe_time, Gval_per_s(bytes[k], pipe_time));
		}*/
	}
		
	lprintf(0,"\n Hop Tranfer Map:\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "  %2d  |", deidxize(d2));
	lprintf(0, "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "-------");
	lprintf(0, "\n");
	for (int d1 = 0; d1 < LOC_NUM; d1++){
		lprintf(0, "%2d | ", deidxize(d1));
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			lprintf(0, "%4d | ", timer_ctr[d1][d2]);
		}
		lprintf(0, "\n");
	}

	lprintf(0,"\n Hop Tranfer Map Achieved Bandwidths (GB/s):\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "  %2d   |", deidxize(d2));
	lprintf(0, "\n   |");
	for (int d2 = 0; d2 < LOC_NUM; d2++)
		lprintf(0, "--------");
	lprintf(0, "\n");
	for (int d1 = 0; d1 < LOC_NUM; d1++){
		lprintf(0, "%2d | ", deidxize(d1));
		for (int d2 = 0; d2 < LOC_NUM; d2++)
			if (timer_ctr[d1][d2]) lprintf(0, "%.2lf | ", link_gbytes_s[d1][d2]/timer_ctr[d1][d2]);
			else lprintf(0, "  -   | ");
		lprintf(0, "\n");
	}
	fclose(fp);
	reseTTEST();
}
#endif
