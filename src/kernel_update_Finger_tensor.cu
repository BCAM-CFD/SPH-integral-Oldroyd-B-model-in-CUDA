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

// The Finger tensor is updated
// This cannot be done in kernel_update_memory, because that is done just by thread 0,
// while this is done for each particle (each thread)
__global__ void kernel_update_Finger_tensor(int*  __restrict__ head_mem,
					    int*  __restrict__ tail_mem,
					    real* __restrict__ Bxx,
					    real* __restrict__ Bxy,
					    real* __restrict__ Byy,
					    real* __restrict__ Bxz,
					    real* __restrict__ Byz,
					    real* __restrict__ Bzz,
					    int*  __restrict__ type) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) return;

  //If it is not a fluid particle, Finger tensor is not used
  if (type[i] != 0) return; 
  
  // The memory position j of the particle i is in i * max_steps_memory + j
  // We only have to add the Finger tensor of the most recent time, j = last_pos,
  // so we have just to initialize N * last_pos + i
  int last_pos = head_mem[0];
  int i_finger = N * last_pos + i;
  Bxx[i_finger] = 1.0;
  Bxy[i_finger] = 0.0;
  Byy[i_finger] = 1.0;
  if (dim == 3) {
    Bxz[i_finger] = 0.0;
    Byz[i_finger] = 0.0;
    Bzz[i_finger] = 1.0;
  }

}
