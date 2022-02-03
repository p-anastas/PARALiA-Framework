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
#ifndef UNIHELPER_LOCKFREE_ENABLE
int unihelper_lock = 0;
#endif

inline void get_lock(){
#ifndef UNIHELPER_LOCKFREE_ENABLE
	while(__sync_lock_test_and_set (&unihelper_lock, 1));
#endif
	;
}
inline void release_lock(){
#ifndef UNIHELPER_LOCKFREE_ENABLE
	__sync_lock_release(&unihelper_lock);
#endif
	;
}

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
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);

	cqueue_backend_data = malloc(sizeof(cublasHandle_t));
	massert(CUBLAS_STATUS_SUCCESS == cublasCreate((cublasHandle_t*) cqueue_backend_data),
		"CommandQueue::CommandQueue: cublasCreate failed\n");
	massert(CUBLAS_STATUS_SUCCESS == cublasSetStream(*((cublasHandle_t*) cqueue_backend_data), stream),
		"cublasSetStream failed\n");
}

CommandQueue::~CommandQueue()
{
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaError_t err = cudaStreamSynchronize(stream);
	massert(cudaSuccess == err, "CommandQueue::CommandQueue - cudaStreamSynchronize: %s\n", cudaGetErrorString(err));
	err = cudaStreamDestroy(stream);
	massert(cudaSuccess == err, "CommandQueue::CommandQueue - cudaStreamDestroy: %s\n", cudaGetErrorString(err));
	free(cqueue_backend_ptr);
	cublasHandle_t handle = *((cublasHandle_t*) cqueue_backend_data);
	massert(CUBLAS_STATUS_SUCCESS == cublasDestroy(handle),
		"CommandQueue::CommandQueue - cublasDestroy(handle) failed\n");
	return;
}

void CommandQueue::sync_barrier()
{
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaError_t err = cudaStreamSynchronize(stream);
	massert(cudaSuccess == err, "CommandQueue::sync_barrier - %s\n", cudaGetErrorString(err));
}

void CommandQueue::add_host_func(void* func, void* data){
	get_lock();
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaError_t err = cudaLaunchHostFunc(stream, (cudaHostFn_t) func, data);
	massert(cudaSuccess == err, "CommandQueue::add_host_func - %s\n", cudaGetErrorString(err));
	release_lock();
}

void CommandQueue::wait_for_event(Event_p Wevent)
{
	if (Wevent->query_status() == CHECKED) return;
	get_lock();
	cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);
	cudaEvent_t cuda_event= *(cudaEvent_t*) Wevent->event_backend_ptr;
	cudaError_t err = cudaStreamWaitEvent(stream, cuda_event, 0); // 0-only parameter = future NVIDIA masterplan?
	massert(cudaSuccess == err, "CommandQueue::wait_for_event - %s\n", cudaGetErrorString(err));
	release_lock();
}

/*****************************************************/
/// Event class functions. TODO: Do status = .. commands need lock?
Event::Event()
{
	get_lock();
	event_backend_ptr = malloc(sizeof(cudaEvent_t));
	int dev_id;  cudaGetDevice(&dev_id);
	Event_num_device[dev_id]++;
	id = Event_num_device[dev_id];
	cudaError_t err = cudaEventCreate(( cudaEvent_t*) event_backend_ptr);
	status = UNRECORDED;
	massert(cudaSuccess == err, "Event::Event - %s\n", cudaGetErrorString(err));
	release_lock();
}

Event::~Event()
{
	sync_barrier();
	get_lock();
	int dev_id;  cudaGetDevice(&dev_id);
	Event_num_device[dev_id]--;
	cudaError_t err = cudaEventDestroy(*(( cudaEvent_t*) event_backend_ptr));
	free(event_backend_ptr);
	massert(cudaSuccess == err, "Event::~Event - %s\n", cudaGetErrorString(err));
	release_lock();
}

void Event::sync_barrier()
{
	get_lock();
	if (status != CHECKED){
		if (status == UNRECORDED){
			warning("Event::sync_barrier: Tried to sync unrecorded event\n");
			return;
		}
		cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
		cudaError_t err = cudaEventSynchronize(cuda_event);
		if (status == RECORDED) status = CHECKED;
		massert(cudaSuccess == err, "Event::sync_barrier - %s\n", cudaGetErrorString(err));
	}
	release_lock();
}

void Event::record_to_queue(CQueue_p Rr){
	get_lock();
	if (Rr == NULL) status = CHECKED;
	else{
		if (status != UNRECORDED){
			warning("Event::record_to_queue: Recording %s event\n", print_event_status(status));
		}
		cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
		cudaStream_t stream = *((cudaStream_t*) Rr->cqueue_backend_ptr);
		cudaError_t err = cudaEventRecord(cuda_event, stream);
		status = RECORDED;
		massert(cudaSuccess == err, "Event::record_to_queue - %s\n", cudaGetErrorString(err));
	}
	release_lock();
}

event_status Event::query_status(){
	get_lock();
	if (status != CHECKED){
		cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
		cudaError_t err = cudaEventQuery(cuda_event);

		if (err == cudaSuccess && (status == UNRECORDED ||  status == COMPLETE));
		else if (err == cudaSuccess && status == RECORDED) status = COMPLETE;
		else if (err == cudaErrorNotReady && status == RECORDED);
		else if (err == cudaErrorNotReady && status == UNRECORDED){
			// this should not happen in a healthy implementation
			warning("Event::query_status: cudaErrorNotReady with status == UNRECORDED should not happen\n");
			status = RECORDED;
		}
		else if (err == cudaSuccess &&  status == CHECKED)
			// TODO: This should not happen in a healthy locked update scenario.
			// But it does since no locking yet. Not sure of its effects.
			warning("Event::query_status: cudaSuccess with status == CHECKED should not happen\n");
		else error("Event::query_status - %s, status=%s\n", cudaGetErrorString(err), print_event_status(status));
	}
	release_lock();
	return status;
}

void Event::checked(){
	get_lock();
	if (status == COMPLETE) status = CHECKED;
	else error("Event::checked(): error event was %s,  not COMPLETE()\n", print_event_status(status));
	release_lock();
}

void Event::reset(){
	get_lock();
	status = UNRECORDED;
	release_lock();
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
