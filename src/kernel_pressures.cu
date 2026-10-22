/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

// Calculation of the pressure with an equation of state
__global__ void kernel_pressures(real* __restrict__ press,
				 real* __restrict__ dens,
				 real* __restrict__ mass,
				 real* __restrict__ inv_dens,
				 real* __restrict__ stressxx_mem,
				 real* __restrict__ stressxy_mem,
				 real* __restrict__ stressyy_mem,
				 real* __restrict__ stressxz_mem,
				 real* __restrict__ stressyz_mem,
				 real* __restrict__ stresszz_mem,
				 real* __restrict__ pressxx_over_dens_sq,
				 real* __restrict__ pressxy_over_dens_sq,
				 real* __restrict__ pressyy_over_dens_sq,
				 real* __restrict__ pressxz_over_dens_sq,
				 real* __restrict__ pressyz_over_dens_sq,
				 real* __restrict__ presszz_over_dens_sq,
				 int*  __restrict__ type) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) return;
  //If it is not a fluid particle, Finger tensor is not calculated  
  if (type[i] != 0) return; 

  real press_i = csq * (mass[i] * dens[i] - rho0);
  press[i] = press_i;

  real inv_dens_i    = inv_dens[i];
  real inv_dens_i_sq = inv_dens_i * inv_dens_i;  

  // We consider stress tensor as symmetric
  pressxx_over_dens_sq[i] = (press_i - stressxx_mem[i]) * inv_dens_i_sq;
  pressxy_over_dens_sq[i] = - stressxy_mem[i] * inv_dens_i_sq;
  pressyy_over_dens_sq[i] = (press_i - stressyy_mem[i]) * inv_dens_i_sq;
  if (dim == 3) {
    pressxz_over_dens_sq[i] = - stressxz_mem[i] * inv_dens_i_sq;
    pressyz_over_dens_sq[i] = - stressyz_mem[i] * inv_dens_i_sq;
    presszz_over_dens_sq[i] = (press_i - stresszz_mem[i]) * inv_dens_i_sq;    
  }
    
  
}
