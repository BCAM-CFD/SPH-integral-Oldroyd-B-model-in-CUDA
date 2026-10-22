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

// Function to calculate force of each particle.
// and density and pressure too.
int class_system::calculate_forces(dim3 numBlocks,
				   dim3 threadsPerBlock,
				   real* k_x,
				   real* k_y,
				   real* k_z,
				   real* k_vx,
				   real* k_vy,
				   real* k_vz,
				   real* k_fx,
				   real* k_fy,
				   real* k_fz,
				   real* k_gradvxx,
				   real* k_gradvxy,
				   real* k_gradvxz,
				   real* k_gradvyx,
				   real* k_gradvyy,
				   real* k_gradvyz,
				   real* k_gradvzx,
				   real* k_gradvzy,
				   real* k_gradvzz,
				   real* k_Bxx,
				   real* k_Bxy,
				   real* k_Byy,
				   real* k_Bxz,
				   real* k_Byz,
				   real* k_Bzz,
				   real* k_dBxx_dt,
				   real* k_dBxy_dt,
				   real* k_dByy_dt,
				   real* k_dBxz_dt,
				   real* k_dByz_dt,
				   real* k_dBzz_dt,
				   int*  k_mem_Ntimes,
				   real* k_mem_time,
				   int*  k_head_mem,
				   int*  k_tail_mem,
				   real* k_press,
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
				   real* k_dens,
				   real* k_inv_dens,				   
				   real* k_mass,				    
				   int*  k_particle_index,
				   int*  k_cell_start,
				   int*  k_cell_end,
				   int*  k_type,
				   real* k_fx_wall,
				   real* k_fy_wall,
				   real* k_fz_wall,
				   real* k_coll_x,
				   real* k_coll_y,
				   real* k_coll_z,
				   real* k_coll_vx,
				   real* k_coll_vy,
				   real* k_coll_vz,
				   real* k_coll_omegax,
				   real* k_coll_omegay,
				   real* k_coll_omegaz,
				   real* k_fx_colloid,
				   real* k_fy_colloid,
				   real* k_fz_colloid,
				   real* k_tx_colloid,
				   real* k_ty_colloid,
				   real* k_tz_colloid,
				   int*  k_colloids_list,
				   int*  k_colloids_start,
				   int*  k_coll_index,
				   int*  k_coll_cell_start,
				   int*  k_coll_cell_end,
				   float4* k_posf,
				   real* k_M_integration) {

  cudaError_t cuda_err;
  int error;

  //---- Positions in float ----
  kernel_build_posf<<<numBlocks, threadsPerBlock>>>(k_x, k_y, k_z, k_type, k_particle_index, k_posf);

  //---- Densities are calculated ----
  error = calculate_density(numBlocks, threadsPerBlock,
			    k_x, k_y, k_z, k_dens, k_inv_dens,
			    k_particle_index, k_cell_start, k_cell_end, k_posf);
  if (error != 0)
    return 1;

  //----- M_integration values are calculated before calculating stress with numerical
  //      integration -----
  kernel_M_integration<<<numBlocks, threadsPerBlock>>>(k_M_integration, k_mem_time);  
  //----- Stress is calculated from memory function -----
  kernel_stress_mem<<<numBlocks, threadsPerBlock>>>(k_stressxx_mem, k_stressxy_mem,
						    k_stressyy_mem, k_stressxz_mem,
						    k_stressyz_mem, k_stresszz_mem,
						    k_Bxx, k_Bxy, k_Byy,
						    k_Bxz, k_Byz, k_Bzz,
						    k_mem_Ntimes, k_mem_time,
						    k_head_mem, k_tail_mem, k_type,
						    k_M_integration);
  // cudaDeviceSynchronize();  // We require the kernel to end to continue

  //---- Pressure is calculated ----
  error = calculate_pressure(numBlocks, threadsPerBlock, k_press, k_dens, k_mass,
			     k_inv_dens, k_stressxx_mem, k_stressxy_mem,
			     k_stressyy_mem, k_stressxz_mem,
			     k_stressyz_mem, k_stresszz_mem,
			     k_pressxx_over_dens_sq, k_pressxy_over_dens_sq,
			     k_pressyy_over_dens_sq, k_pressxz_over_dens_sq,
			     k_pressyz_over_dens_sq, k_presszz_over_dens_sq, k_type);
  if (error != 0)
    return 1;
      
  //----- forces are calculated ------
  kernel_forces_and_vel_gradients<<<numBlocks, threadsPerBlock>>>(k_x, k_y, k_z,
								  k_vx, k_vy, k_vz,
								  k_fx, k_fy, k_fz,
								  k_gradvxx, k_gradvxy,
								  k_gradvxz, k_gradvyx,
								  k_gradvyy, k_gradvyz,
								  k_gradvzx, k_gradvzy,
								  k_gradvzz,
								  k_pressxx_over_dens_sq,
								  k_pressxy_over_dens_sq,
								  k_pressyy_over_dens_sq,
								  k_pressxz_over_dens_sq,
								  k_pressyz_over_dens_sq,
								  k_presszz_over_dens_sq,
								  k_inv_dens, k_particle_index,
								  k_cell_start, k_cell_end, k_type,
								  k_coll_x, k_coll_y, k_coll_z,
								  k_coll_vx, k_coll_vy, k_coll_vz,
								  k_coll_omegax, k_coll_omegay,
								  k_coll_omegaz, k_posf);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf("Error in kernel_forces: %s\n", cudaGetErrorString(cuda_err));
    return 1;
  }    
  // cudaDeviceSynchronize();  // We require the kernel to end to continue
  

  //--- The time derivative of the Finger tensor is necessary, and it is calculated
  //    in kernel_move_sys_part2_and_Finger_der.cu

  //---- External forces are calculated ----
  if (ext_force_type != 0) {
    error = calculate_external_forces(numBlocks, threadsPerBlock,
				      k_x, k_y, k_z,
				      k_fx, k_fy, k_fz, k_mass, k_type);
    if (error != 0)
      return 1; // End of program
  }

  //---- Right now, forces on walls are not calculated here, but at system_print_output----
  
  //---- Forces and torques on colloids are calculated ----
  if (N_colloids > 0) {
    error = calculate_force_colloids(numBlocks, threadsPerBlock,
				     k_coll_x, k_coll_y, k_coll_z,
				     k_fx_colloid, k_fy_colloid, k_fz_colloid,
				     k_tx_colloid, k_ty_colloid, k_tz_colloid,
				     k_x, k_y, k_z, k_fx, k_fy, k_fz,
				     k_colloids_list, k_colloids_start,
				     k_coll_index, k_coll_cell_start, k_coll_cell_end);
    if (error != 0)
      return 1; // End of program
  }  

  return 0;

}
