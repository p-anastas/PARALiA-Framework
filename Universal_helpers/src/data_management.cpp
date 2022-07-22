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
//#include "backend_wrappers.hpp"

//#define SPLIT_2D_ROWISE
void FasTCoCoMemcpy2DAsync(link_road_p roadMap, size_t rows, size_t cols, short elemSize){
	if(roadMap->hop_num < 2) error("FasTCoCoMemcpy2DAsync: Cannot copy with less than 2 locations\n");

	int buffer_bw_overlap = 8;
	Event_p step_events[roadMap->hop_num][buffer_bw_overlap];
	for(int uid_ctr = 0; uid_ctr < roadMap->hop_num - 1; uid_ctr++){
#ifdef SPLIT_2D_ROWISE
		size_t local_rows = rows/buffer_bw_overlap;
#else
		size_t local_cols = cols/buffer_bw_overlap;
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
	}
}
