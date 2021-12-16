///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <cstdio>
#include <typeinfo>
#include <float.h>
#include <curand.h>

#include "backend_wrappers.hpp"

int Event_num_device[128] = {0};

/*****************************************************/
/// Event Status-related functions

const char* print_event_status(event_status in_status){
	switch(in_status){
		case(UNRECORDED):
			return "UNRECORDED";
		case(RECORDED):
			return "RECORDED";
		case(COMPLETE):
			return "COMPLETE";
		case(CHECKED):
			return "CHECKED";
		case(GHOST):
			return "GHOST";
		default:
			error("print_event_status: Unknown state\n");
	}
}

/*****************************************************/
/// Command queue class functions
CommandQueue::CommandQueue()
{
	cqueue_backend_ptr = malloc(sizeof(cudaStream_t));
	cudaError_t err = cudaStreamCreate((cudaStream_t*) cqueue_backend_ptr);
	massert(cudaSuccess == err, "CommandQueue::CommandQueue - %s\n", cudaGetErrorString(err));
}

void CommandQueue::sync_barrier()
{
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaError_t err = cudaStreamSynchronize(stream);
	massert(cudaSuccess == err, "CommandQueue::sync_barrier - %s\n", cudaGetErrorString(err));
}

void CommandQueue::wait_for_event(Event_p Wevent)
{
	if (Wevent->query_status() == CHECKED) return;
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaEvent_t cuda_event= *(cudaEvent_t*) Wevent->event_backend_ptr;
	cudaError_t err = cudaStreamWaitEvent(stream, cuda_event, 0); // 0-only parameter = future NVIDIA masterplan?
	massert(cudaSuccess == err, "CommandQueue::wait_for_event - %s\n", cudaGetErrorString(err));
}

/*****************************************************/
/// Event class functions. TODO: Do status = .. commands need lock?
Event::Event()
{
	event_backend_ptr = malloc(sizeof(cudaEvent_t));
	int dev_id;  cudaGetDevice(&dev_id);
	Event_num_device[dev_id]++;
	id = Event_num_device[dev_id];
	cudaError_t err = cudaEventCreate(( cudaEvent_t*) event_backend_ptr);
	status = UNRECORDED;
	massert(cudaSuccess == err, "Event::Event - %s\n", cudaGetErrorString(err));
}

void Event::sync_barrier()
{
	if (status == CHECKED) return;
	else if (status == UNRECORDED){
		warning("Event::sync_barrier: Tried to sync unrecorded event\n");
		return;
	}
	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
	cudaError_t err = cudaEventSynchronize(cuda_event);
	if (status == RECORDED) status = COMPLETE;
	massert(cudaSuccess == err, "Event::sync_barrier - %s\n", cudaGetErrorString(err));
}

void Event::record_to_queue(CQueue_p Rr){
	if (Rr == NULL){
		status = CHECKED;
		return;
	}
	else if (status != UNRECORDED){
		warning("Event::record_to_queue: Recording %s event\n", print_event_status(status));
	}
	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
	cudaStream_t stream = *((cudaStream_t*) Rr->cqueue_backend_ptr);
	cudaError_t err = cudaEventRecord(cuda_event, stream);
	status = RECORDED;
	massert(cudaSuccess == err, "Event::record_to_queue - %s\n", cudaGetErrorString(err));
}

event_status Event::query_status(){
	if (status == CHECKED) return status;
	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
	cudaError_t err = cudaEventQuery(cuda_event);
	if (err == cudaSuccess && (status == UNRECORDED ||  status == COMPLETE)) return status;
	else if (err == cudaSuccess && status == RECORDED){ // Event has finished but not synched yet!
		status = COMPLETE;
		return status;
	}
	else if (err == cudaErrorNotReady && status == RECORDED) return status;
	else if (err == cudaErrorNotReady && status == UNRECORDED){
		// this should not happen in a healthy implementation
		warning("Event::query_status: cudaErrorNotReady with status == UNRECORDED should not happen\n");
		status = RECORDED;
		return status;
	}
	else error("Event::query_status - %s, status=%s\n", cudaGetErrorString(err), print_event_status(status));
}

void Event::checked(){
	if (status == COMPLETE) status = CHECKED;
	else error("Event::checked(): error event was %s,  not COMPLETE()\n", print_event_status(status));
}

void Event::reset(){
	status = UNRECORDED;
}

/*****************************************************/
/// Event-based timer class functions

Event_timer::Event_timer() {
  Event_start = new Event();
  Event_stop = new Event();
  time_ms = 0;
}

void Event_timer::start_point(CQueue_p start_queue)
{
	Event_start->record_to_queue(start_queue);
	//cudaStream_t stream = *((cudaStream_t*) start_queue->cqueue_backend_ptr);
	//cudaEvent_t cuda_event = *(cudaEvent_t*) Event_start->event_backend_ptr;
	//cudaEventRecord(cuda_event, stream);
}

void Event_timer::stop_point(CQueue_p stop_queue)
{
	Event_stop->record_to_queue(stop_queue);
	//cudaStream_t stream = *((cudaStream_t*) stop_queue->cqueue_backend_ptr);
	//cudaEvent_t cuda_event = *(cudaEvent_t*) Event_stop->event_backend_ptr;
	//cudaEventRecord(cuda_event, stream);
}

double Event_timer::sync_get_time()
{
	float temp_t;
	//cudaEvent_t cuda_event_start = *(cudaEvent_t*) Event_start->event_backend_ptr;
	//cudaEvent_t cuda_event_stop = *(cudaEvent_t*) Event_stop->event_backend_ptr;
	//cudaEventSynchronize(cuda_event_start);
	//cudaEventSynchronize(cuda_event_stop);
	Event_start->sync_barrier();
	Event_stop->sync_barrier();
	cudaEvent_t cuda_event_start = *(cudaEvent_t*) Event_start->event_backend_ptr;
	cudaEvent_t cuda_event_stop = *(cudaEvent_t*) Event_stop->event_backend_ptr;
	cudaEventElapsedTime(&temp_t, cuda_event_start, cuda_event_stop);
	time_ms = (double) temp_t;
	return time_ms;
}

/*****************************************************/
