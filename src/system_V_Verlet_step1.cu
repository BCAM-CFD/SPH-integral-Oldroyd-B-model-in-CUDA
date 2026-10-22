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

// Function to do the first step of the Velocity Verlet algorithm
int class_system::V_Verlet_step1(dim3 numBlocks,
				 dim3 threadsPerBlock,
				 real* k_x,
				 real* k_y,
				 real* k_z,
				 real* k_vx,
				 real* k_vy,
				 real* k_vz,
				 real* k_Bxx,
				 real* k_Bxy,
				 real* k_Byy,
				 real* k_Bxz,
				 real* k_Byz,
				 real* k_Bzz,
				 real* k_fx,
				 real* k_fy,
				 real* k_fz,
				 real* k_dBxx_dt,
				 real* k_dBxy_dt,
				 real* k_dByy_dt,
				 real* k_dBxz_dt,
				 real* k_dByz_dt,
				 real* k_dBzz_dt,				 
				 real* k_mass,
				 int*  k_type,
				 int*  k_mem_Ntimes,
				 int*  k_head_mem,
				 int*  k_tail_mem,				 
				 real* k_coll_x,
				 real* k_coll_y,
				 real* k_coll_z,
				 real* k_coll_vx,
				 real* k_coll_vy,
				 real* k_coll_vz,
				 real* k_coll_omegax,
				 real* k_coll_omegay,
				 real* k_coll_omegaz,
				 real* k_coll_theta,
				 real* k_coll_quat0,
				 real* k_coll_quatx,
				 real* k_coll_quaty,
				 real* k_coll_quatz,
				 real* k_fx_colloid,
				 real* k_fy_colloid,
				 real* k_fz_colloid,
				 real* k_tx_colloid,
				 real* k_ty_colloid,
				 real* k_tz_colloid,				 
				 real* k_x_center,
				 real* k_y_center,
				 real* k_z_center,
				 int*  k_colloids_list,
				 int*  k_colloids_start,
				 int   coll_move) {

  cudaError_t cuda_err;

  //--- Fluid and walls particles are moved ---  
  kernel_move_sys_VV_part1<<<numBlocks, threadsPerBlock>>>(k_x, k_y, k_z,
							   k_vx, k_vy, k_vz,
							   k_Bxx, k_Bxy, k_Byy,
							   k_Bxz, k_Byz, k_Bzz,
							   k_fx, k_fy, k_fz,
							   k_dBxx_dt, k_dBxy_dt,
							   k_dByy_dt, k_dBxz_dt,
							   k_dByz_dt, k_dBzz_dt,
							   k_mass, k_type,
							   k_mem_Ntimes,
							   k_head_mem, k_tail_mem);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf("Error in kernel_move_sys: %s\n", cudaGetErrorString(cuda_err));
    return 1;
  }

  //--- Particles of colloids are moved ---
  if (N_colloids > 0)
    if (coll_move == 1) {
      kernel_move_colloids_VV_part1<<<numBlocks, threadsPerBlock>>>(k_coll_x, k_coll_y,
								    k_coll_z, k_coll_vx,
								    k_coll_vy, k_coll_vz,
								    k_coll_omegax,
								    k_coll_omegay,
								    k_coll_omegaz,
								    k_coll_theta,
								    k_coll_quat0,
								    k_coll_quatx,
								    k_coll_quaty,
								    k_coll_quatz,
								    k_fx_colloid,
								    k_fy_colloid,
								    k_fz_colloid,
								    k_tx_colloid,
								    k_ty_colloid,
								    k_tz_colloid,
								    k_x, k_y, k_z,
								    k_x_center,
								    k_y_center,
								    k_z_center,
								    k_colloids_list,
								    k_colloids_start);
      cuda_err = cudaGetLastError();
      if (cuda_err != cudaSuccess) {
	printf("Error in kernel_colloids_VV_part1: %s\n", cudaGetErrorString(cuda_err));
	return 1;
      }
    }
  
  // cudaDeviceSynchronize();  // We require the kernel to end to continue    

  return 0;

}
