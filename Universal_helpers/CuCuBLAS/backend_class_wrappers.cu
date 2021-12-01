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
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaEvent_t cuda_event= *(cudaEvent_t*) Wevent->event_backend_ptr;
	cudaError_t err = cudaStreamWaitEvent(stream, cuda_event, 0); // 0-only parameter = future NVIDIA masterplan?
	massert(cudaSuccess == err, "CommandQueue::wait_for_event - %s\n", cudaGetErrorString(err));
}

/*****************************************************/
/// Event class functions
Event::Event()
{
	event_backend_ptr = malloc(sizeof(cudaEvent_t));
	cudaError_t err = cudaEventCreate(( cudaEvent_t*) event_backend_ptr);
	massert(cudaSuccess == err, "Event::Event - %s\n", cudaGetErrorString(err));
}

void Event::sync_barrier()
{
	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
	cudaError_t err = cudaEventSynchronize(cuda_event);
	massert(cudaSuccess == err, "Event::sync_barrier - %s\n", cudaGetErrorString(err));
}

void Event::record_to_queue(CQueue_p Rr){
	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
	cudaStream_t stream = *((cudaStream_t*) Rr->cqueue_backend_ptr);
	cudaError_t err = cudaEventRecord(cuda_event, stream);
	massert(cudaSuccess == err, "Event::record_to_queue - %s\n", cudaGetErrorString(err));
}

short Event::is_complete(){
	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
	cudaError_t err = cudaEventQuery(cuda_event);
	if (err == cudaSuccess) return 1;
	if (err == cudaErrorNotReady) return 0;
	else error("Event::is_complete - %s\n", cudaGetErrorString(err));
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
