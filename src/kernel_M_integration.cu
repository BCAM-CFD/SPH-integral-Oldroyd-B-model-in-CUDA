/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

// The memory function M(0, t0) of every memory location is calculated.
// It is the same for all the particles, so it is calculated only once per time step
// here, instead of in every thread of the numerical integration in kernel_stress_mem.
// Note that it calculates M_integration for all positions of the mem_time array,
// no matter they are used or not.
__global__ void kernel_M_integration(real* __restrict__ M_integration,
				     real* __restrict__ mem_time) {
  int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k >= max_steps_memory) return;

  M_integration[k] = kernel_memory_function(0.0, mem_time[k]);
}
