/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include "kernel_functions.h"
#include <stdio.h>

// Function to update several arrays related to memory
int class_system::update_memory(dim3  numBlocks,
				dim3  threadsPerBlock,
				real* k_Bxx,
				real* k_Bxy,
				real* k_Byy,
				real* k_Bxz,
				real* k_Byz,
				real* k_Bzz,
				real* k_mem_time,
				int*  k_mem_Ntimes,
				real* k_time_subs,
				int*  k_time_subs_filter,
				int*  k_marked_memory_times,
				int*  k_Nmarked,
				int*  k_head_mem,
				int*  k_tail_mem,				
				int*  k_Nfree_mem,
				int*  k_free_mem,
				int*  k_type) {
  cudaError_t cuda_err;

  //--- memory array is updated ---
  kernel_update_memory_array<<<numBlocks, threadsPerBlock>>>(k_mem_time,
							     k_mem_Ntimes,
							     k_time_subs,
							     k_time_subs_filter,
							     k_marked_memory_times,
							     k_Nmarked,
							     k_head_mem,
							     k_tail_mem,
							     k_Nfree_mem,
							     k_free_mem);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf("Error in kernel_update_memory_array: %s\n", cudaGetErrorString(cuda_err));
    return 1;
  }      
  // cudaDeviceSynchronize();  // We require the kernel to end to continue

  //--- Finger tensor is updated ---
  kernel_update_Finger_tensor<<<numBlocks, threadsPerBlock>>>(k_head_mem,
							      k_tail_mem,
							      k_Bxx,
							      k_Bxy,
							      k_Byy,
							      k_Bxz,
							      k_Byz,
							      k_Bzz,
							      k_type);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf("Error in kernel_update_Finger_tensor: %s\n", cudaGetErrorString(cuda_err));
    return 1;
  }      
  // cudaDeviceSynchronize();  // We require the kernel to end to continue


  return 0;  // End of function

}
