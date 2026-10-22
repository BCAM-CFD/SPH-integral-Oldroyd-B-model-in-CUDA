/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <assert.h>
#include <stdio.h>

// The memory array is updated
__global__ void kernel_update_memory_array(real* __restrict__ mem_time,
					   int*  __restrict__ mem_Ntimes,
					   real* __restrict__ time_subs,
					   int*  __restrict__ time_subs_filter,
					   int*  __restrict__ marked_memory_times,
					   int*  __restrict__ Nmarked,
					   int*  __restrict__ head_mem,
					   int*  __restrict__ tail_mem,
					   int*  __restrict__ Nfree_mem,
					   int*  __restrict__ free_mem) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 0) return;  // This is done only by the thread 0

  if (mem_Ntimes[0] == max_steps_memory) {
    printf("kernel update memory array error: The maximum size of the memory array was exceded \n");
     assert(0);
   }
  else { //-- First we marked memory locations to be erased --
    int mem_time_pos;
    int i_filter = 0;    
    real current_time;
    Nmarked[0] = 0;
    //--- We consider that the first time is not removed ---
    if (head_mem[0] >= 0) { //It is < 0 in the first step
      int current_pos = tail_mem[head_mem[0]];  // .i.e. any time has been already stored
      for (int j = 0; j < mem_Ntimes[0]-2; ++j) {
	mem_time_pos = tail_mem[current_pos]; // tail_mem points to the location of current_time
	current_time = mem_time[mem_time_pos];
	// if current time crosses a limit time (given by times_subs[filter])
	if (current_time > time_subs[i_filter] &&
	    current_time - dt < time_subs[i_filter]) {
	  if (time_subs_filter[i_filter] != 0) {
	    // In this case, this memory time is marked to be deleted 
	    marked_memory_times[Nmarked[0]] = mem_time_pos; // It is filtered (marked to be erased)  
	    Nmarked[0] = Nmarked[0] + 1;
	    // The filter changes only if it is not the last filter
	    if (i_filter != N_subint - 1) 	    
	      if (time_subs_filter[i_filter] != freq_filter - 1) 
		time_subs_filter[i_filter] = time_subs_filter[i_filter] + 1;
	      else
		time_subs_filter[i_filter] = 0;
	  }
	  else // if time_subs_filter[i_filter] == 0
	    time_subs_filter[i_filter] = time_subs_filter[i_filter] + 1; //-- The filter changes
	  i_filter = i_filter + 1; // if checked time is below limit time, go to the next limit time
	}
	else if (current_time < time_subs[i_filter])
	  i_filter = i_filter + 1; // if checked time is below limit time, go to the next limit time
	current_pos = tail_mem[current_pos]; // current_time is updated
      }
    }
  }

  // We remove the locations marked above
  if (Nmarked[0] > 0) {
    int j_to_remove;
    for (int j = 0; j < Nmarked[0]; ++j) {
      j_to_remove = marked_memory_times[j];
      for (int k = 0; k < max_steps_memory; ++k)
  	if (tail_mem[k] == j_to_remove) {
  	  tail_mem[k]           = tail_mem[j_to_remove]; // tail is replaced
  	  tail_mem[j_to_remove] = -1;
  	}
      mem_Ntimes[0]              = mem_Ntimes[0] - 1;
      Nfree_mem[0]               = Nfree_mem[0] + 1;
      free_mem[Nfree_mem[0] - 1] = j_to_remove; // This location is free now
    }
  }

  // We add the most recent memory time
  mem_Ntimes[0]     = mem_Ntimes[0] + 1;
  int new_pos       = free_mem[Nfree_mem[0] - 1];
  mem_time[new_pos] = 0.0;
  tail_mem[new_pos] = head_mem[0];
  head_mem[0]       = new_pos;
  Nfree_mem[0]      = Nfree_mem[0] - 1;

  // Stored times are updated, substracting dt
  int pos0       = head_mem[0];
  mem_time[pos0] = mem_time[pos0] - dt;
  int last_pos   = tail_mem[pos0];
  for (int j = 1; j < mem_Ntimes[0]; ++j) {
    mem_time[last_pos] = mem_time[last_pos] - dt;
    last_pos           = tail_mem[last_pos];
  }
  
}
