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

// Function to calculate the pressure of each particle
int class_system::calculate_pressure(dim3 numBlocks,
				     dim3 threadsPerBlock,
				     real* k_press,
				     real* k_dens,
				     real* k_mass,
				     real* k_inv_dens,
				     real* k_stressxx_mem,
				     real* k_stressxy_mem,
				     real* k_stressyy_mem,
				     real* k_stressxz_mem,
				     real* k_stressyz_mem,
				     real* k_stresszz_mem,
				     real* k_pressxx_over_dens_sq,
				     real* k_pressxy_over_dens_sq,
				     real* k_pressyy_over_dens_sq,
				     real* k_pressxz_over_dens_sq,
				     real* k_pressyz_over_dens_sq,
				     real* k_presszz_over_dens_sq,				     
				     int*  k_type)
{

    cudaError_t cuda_err;
    
    kernel_pressures<<<numBlocks, threadsPerBlock>>>(k_press, k_dens, k_mass, k_inv_dens,
						     k_stressxx_mem, k_stressxy_mem,
						     k_stressyy_mem, k_stressxz_mem,
						     k_stressyz_mem, k_stresszz_mem,
						     k_pressxx_over_dens_sq, k_pressxy_over_dens_sq,
						     k_pressyy_over_dens_sq, k_pressxz_over_dens_sq,
						     k_pressyz_over_dens_sq, k_presszz_over_dens_sq,
						     k_type);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      printf("Error in kernel_pressures: %s\n", cudaGetErrorString(cuda_err));
      return 1;
    }      
    // cudaDeviceSynchronize();  // We require the kernel to end to continue

    return 0;

}
