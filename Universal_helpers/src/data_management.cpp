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
double timers[LOC_NUM][LOC_NUM][2][10000];
long long bytes[LOC_NUM][LOC_NUM][10000];
int timer_ctr[LOC_NUM][LOC_NUM] = {{0}};

void reseTTEST(){
	for(int i = 0; i < LOC_NUM; i++)
		for(int j = 0; j < LOC_NUM; j++){
			for(int k = 0; k < timer_ctr[i][j]; k++){
				bytes[i][j][k] = 0;
				timers[i][j][0][k] = 0;
				timers[i][j][1][k] = 0;
			}
			timer_ctr[i][j] = 0;
		}
}
#endif

//#define SPLIT_2D_ROWISE
void FasTCoCoMemcpy2DAsync(link_road_p roadMap, long int rows, long int cols, short elemSize){
	if(roadMap->hop_num - roadMap->starting_hop < 2) error("FasTCoCoMemcpy2DAsync: Cannot copy with less than 2 locations\n");
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
#ifdef TTEST
			if (timer_ctr[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])] > 10000)
				error("FasTCoCoMemcpy2DAsync(dest = %d, src = %d) exeeded 10000 transfers in TTEST Mode\n",
					roadMap->hop_uid_list[uid_ctr + 1], roadMap->hop_uid_list[uid_ctr]);
			bytes[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]
			[timer_ctr[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]] += rows*cols*elemSize;
			//roadMap->hop_cqueue_list[uid_ctr]->add_host_func(
			//	(void*)&CoCoSetTimerAsync, (void*) &(timers[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]
			//	[0][timer_ctr[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]]));
			CoCoSetTimerAsync(&(timers[idxize(roadMap->hop_uid_list[uid_ctr + 1])]
				[idxize(roadMap->hop_uid_list[uid_ctr])][0]
				[timer_ctr[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]]));
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
			CoCoMemcpy2DAsync(roadMap->hop_buf_list[uid_ctr + 1] + buff_offset_dest, roadMap->hop_ldim_list[uid_ctr + 1],
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
	roadMap->hop_cqueue_list[uid_ctr]->add_host_func(
	(void*)&CoCoSetTimerAsync, (void*) &(timers[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]
	[1][timer_ctr[idxize(roadMap->hop_uid_list[uid_ctr + 1])][idxize(roadMap->hop_uid_list[uid_ctr])]++]));
#endif
	}
}

#ifdef TTEST
void HopMemcpyPrint(){
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
		for (int d2 = 0; d2 < LOC_NUM; d2++){
			if(timer_ctr[d1][d2]){
				double total_bw = 0;
				for (int idx = 0; idx < timer_ctr[d1][d2]; idx++) total_bw+= (
					Gval_per_s(bytes[d1][d2][idx], timers[d1][d2][1][idx] - timers[d1][d2][0][idx]));
				lprintf(0, "%04.2lf | ", total_bw/timer_ctr[d1][d2] );
			}
			else lprintf(0, "  -   | ");
		}
		lprintf(0, "\n");
	}
	reseTTEST();
}
#endif
