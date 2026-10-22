/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <math.h>
#include <stdio.h>

// Function to calculate the stress from the memory function
__global__ void kernel_stress_mem(real* __restrict__ stressxx_mem,
				  real* __restrict__ stressxy_mem,
				  real* __restrict__ stressyy_mem,
				  real* __restrict__ stressxz_mem,
				  real* __restrict__ stressyz_mem,
				  real* __restrict__ stresszz_mem, 
				  real* __restrict__ Bxx,
				  real* __restrict__ Bxy,
				  real* __restrict__ Byy,
				  real* __restrict__ Bxz,
				  real* __restrict__ Byz,
				  real* __restrict__ Bzz,				  
				  int*  __restrict__ mem_Ntimes,
				  real* __restrict__ mem_time,
				  int*  __restrict__ head_mem,
				  int*  __restrict__ tail_mem,
				  int*  __restrict__ type,
				  real* __restrict__ M_integration) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) return;

  //If it is not a fluid particle, Finger tensor is not calculated  
  if (type[i] != 0) return;   

  // The calculation is
  // - int_{-infty}^{t} M(t,t') * gamma_[0] dt'
  // where gamma_[0] = Id - B
  // B is the Finger tensor.
  // The integral is calculated with the trapezoidal rule:
  // int f(x)dx = sum 0.5 * (f_i + f_{i+1}) * (x_{i+1} - x)

  real stressxx_i = 0.0;
  real stressxy_i = 0.0;
  real stressyy_i = 0.0;
  real stressxz_i;
  real stressyz_i;
  real stresszz_i;
  if (dim == 3) {
    stressxz_i = 0.0;
    stressyz_i = 0.0;
    stresszz_i = 0.0;
  }

  // The memory position j of the particle i is in j * max_steps_memory + i
  if (dim == 2) {
    if (head_mem[0] >= 0) { //It is < 0 in the first step    
      int last_pos  = head_mem[0];
      int i_finger  = N * last_pos + i;
      real tf       = 0.0;
      real gammaxxf = 0.0;
      real gammaxyf = 0.0;
      real gammayyf = 0.0;
      real Mf       = E/tau; //M(0) 
      real t0       = mem_time[last_pos];
      real gammaxx0 = 1.0 - Bxx[i_finger];
      real gammaxy0 = -Bxy[i_finger];
      real gammayy0 = 1.0 - Byy[i_finger];
      real M0       = kernel_memory_function(0.0, -dt); //M(t0)
      stressxx_i    = stressxx_i + 0.5 * (Mf * gammaxxf + M0 * gammaxx0) * (tf - t0);
      stressxy_i    = stressxy_i + 0.5 * (Mf * gammaxyf + M0 * gammaxy0) * (tf - t0);
      stressyy_i    = stressyy_i + 0.5 * (Mf * gammayyf + M0 * gammayy0) * (tf - t0);
      last_pos      = tail_mem[last_pos];
      for (int j = 1; j < mem_Ntimes[0]; ++j) {
  	i_finger   = N * last_pos + i;
	tf         = t0;
	gammaxxf   = gammaxx0;
	gammaxyf   = gammaxy0;
	gammayyf   = gammayy0;
	Mf         = M0;
  	t0         = mem_time[last_pos];
  	gammaxx0   = 1.0 - Bxx[i_finger];
  	gammaxy0   = -Bxy[i_finger];
  	gammayy0   = 1.0 - Byy[i_finger];
  	M0         = M_integration[last_pos]; //M(t0)
  	stressxx_i = stressxx_i + 0.5 * (Mf * gammaxxf + M0 * gammaxx0) * (tf - t0);
  	stressxy_i = stressxy_i + 0.5 * (Mf * gammaxyf + M0 * gammaxy0) * (tf - t0);
  	stressyy_i = stressyy_i + 0.5 * (Mf * gammayyf + M0 * gammayy0) * (tf - t0);
  	last_pos   = tail_mem[last_pos];
      }
    }
  }
  else { // dim = 3
    if (head_mem[0] >= 0) { //It is < 0 in the first step    
      int last_pos  = head_mem[0];
      int i_finger  = N * last_pos + i;
      real tf       = 0.0;
      real gammaxxf = 0.0;
      real gammaxyf = 0.0;
      real gammayyf = 0.0;
      real gammaxzf = 0.0;
      real gammayzf = 0.0;
      real gammazzf = 0.0;            
      real Mf       = E/tau; //M(0) 
      real t0       = mem_time[last_pos];
      real gammaxx0 = 1.0 - Bxx[i_finger];
      real gammaxy0 = -Bxy[i_finger];
      real gammayy0 = 1.0 - Byy[i_finger];
      real gammaxz0 = - Bxz[i_finger];
      real gammayz0 = - Byz[i_finger];
      real gammazz0 = 1.0 - Bzz[i_finger];      
      real M0       = kernel_memory_function(0.0, -dt); //M(t0)
      stressxx_i    = stressxx_i + 0.5 * (Mf * gammaxxf + M0 * gammaxx0) * (tf - t0);
      stressxy_i    = stressxy_i + 0.5 * (Mf * gammaxyf + M0 * gammaxy0) * (tf - t0);
      stressyy_i    = stressyy_i + 0.5 * (Mf * gammayyf + M0 * gammayy0) * (tf - t0);
      stressxz_i    = stressxz_i + 0.5 * (Mf * gammaxzf + M0 * gammaxz0) * (tf - t0);
      stressyz_i    = stressyz_i + 0.5 * (Mf * gammayzf + M0 * gammayz0) * (tf - t0);
      stresszz_i    = stresszz_i + 0.5 * (Mf * gammazzf + M0 * gammazz0) * (tf - t0);
      last_pos      = tail_mem[last_pos];
      for (int j = 1; j < mem_Ntimes[0]; ++j) {
  	i_finger   = N * last_pos + i;
	tf         = t0;
	gammaxxf   = gammaxx0;
	gammaxyf   = gammaxy0;
	gammayyf   = gammayy0;
	gammaxzf   = gammaxz0;
	gammayzf   = gammayz0;
	gammazzf   = gammazz0;	
	Mf         = M0;
  	t0         = mem_time[last_pos];
  	gammaxx0   = 1.0 - Bxx[i_finger];
  	gammaxy0   = -Bxy[i_finger];
  	gammayy0   = 1.0 - Byy[i_finger];
  	gammaxz0   = -Bxz[i_finger];
  	gammayz0   = -Byz[i_finger];
  	gammazz0   = 1.0 - Bzz[i_finger];	
  	M0         = M_integration[last_pos]; //M(t0)
  	stressxx_i = stressxx_i + 0.5 * (Mf * gammaxxf + M0 * gammaxx0) * (tf - t0);
  	stressxy_i = stressxy_i + 0.5 * (Mf * gammaxyf + M0 * gammaxy0) * (tf - t0);
  	stressyy_i = stressyy_i + 0.5 * (Mf * gammayyf + M0 * gammayy0) * (tf - t0);
  	stressxz_i = stressxz_i + 0.5 * (Mf * gammaxzf + M0 * gammaxz0) * (tf - t0);
  	stressyz_i = stressyz_i + 0.5 * (Mf * gammayzf + M0 * gammayz0) * (tf - t0);
  	stresszz_i = stresszz_i + 0.5 * (Mf * gammazzf + M0 * gammazz0) * (tf - t0);
  	last_pos   = tail_mem[last_pos];
      }
    }    
  }

  // The stress 
  stressxx_mem[i] = -stressxx_i;
  stressxy_mem[i] = -stressxy_i;
  stressyy_mem[i] = -stressyy_i;
  if (dim == 3) {
    stressxz_mem[i] = -stressxz_i;
    stressyz_mem[i] = -stressyz_i;
    stresszz_mem[i] = -stresszz_i;
  }
  
}
