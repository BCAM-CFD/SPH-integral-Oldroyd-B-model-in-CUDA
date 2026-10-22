/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "cuda_runtime.h"

#include <stdio.h>

// Function to copy the system data to the device
void class_system::copy_pointers_to_device(real** k_x,
					   real** k_y,
					   real** k_z,
					   real** k_vx,
					   real** k_vy,
					   real** k_vz,
					   real** k_mass,
					   real** k_fx,
					   real** k_fy,
					   real** k_fz,
					   real** k_dens,
					   real** k_inv_dens,
					   real** k_press,
					   int**  k_particle_cell,
					   int**  k_particle_index,
					   int**  k_cell_start,
					   int**  k_cell_end,
					   real** k_kin_energy,
					   int**  k_type,
					   real** k_pressxx_over_dens_sq,
					   real** k_pressxy_over_dens_sq,
					   real** k_pressyy_over_dens_sq,
					   real** k_pressxz_over_dens_sq,
					   real** k_pressyz_over_dens_sq,
					   real** k_presszz_over_dens_sq,
					   real** k_gradvxx,
					   real** k_gradvxy,
					   real** k_gradvxz,
					   real** k_gradvyx,
					   real** k_gradvyy,
					   real** k_gradvyz,
					   real** k_gradvzx,
					   real** k_gradvzy,
					   real** k_gradvzz,
					   real** k_fx_wall,
					   real** k_fy_wall,
					   real** k_fz_wall,
					   int**  k_walls_list,
					   int**  k_walls_start,
					   int**  k_colloids_list,
					   int**  k_colloids_start,
					   real** k_coll_x,
					   real** k_coll_y,
					   real** k_coll_z,
					   real** k_coll_vx,
					   real** k_coll_vy,
					   real** k_coll_vz,
					   real** k_coll_omegax,
					   real** k_coll_omegay,
					   real** k_coll_omegaz,
					   real** k_coll_theta,
					   real** k_coll_quat0,
					   real** k_coll_quatx,
					   real** k_coll_quaty,
					   real** k_coll_quatz,
					   real** k_x_center,
					   real** k_y_center,
					   real** k_z_center,
					   real** k_fx_colloid,
					   real** k_fy_colloid,
					   real** k_fz_colloid,
					   real** k_tx_colloid,
					   real** k_ty_colloid,
					   real** k_tz_colloid,
					   int**  k_coll_cell,
					   int**  k_coll_index,
					   int**  k_coll_cell_start,
					   int**  k_coll_cell_end,
					   real** k_Bxx,
					   real** k_Bxy,
					   real** k_Byy,
					   real** k_Bxz,
					   real** k_Byz,
					   real** k_Bzz,
					   real** k_dBxx_dt,
					   real** k_dBxy_dt,
					   real** k_dByy_dt,
					   real** k_dBxz_dt,
					   real** k_dByz_dt,
					   real** k_dBzz_dt,
					   real** k_mem_time,
					   real** k_time_subs,
					   int**  k_time_subs_filter,
					   int**  k_mem_Ntimes,
					   int**  k_marked_memory_times,
					   int**  k_Nmarked,
					   int**  k_head_mem,
					   int**  k_tail_mem, 
					   int**  k_Nfree_mem,
					   int**  k_free_mem,
					   real** k_stressxx_mem,
					   real** k_stressxy_mem,
					   real** k_stressyy_mem,
					   real** k_stressxz_mem,
					   real** k_stressyz_mem,
					   real** k_stresszz_mem,
					   float4** k_posf,
					   real** k_M_integration) {
  
  //------- Variables to copy -------------
  cudaMalloc(k_x             , N * sizeof(real));
  cudaMalloc(k_y             , N * sizeof(real));
  cudaMalloc(k_z             , N * sizeof(real));
  cudaMalloc(k_vx            , N * sizeof(real));
  cudaMalloc(k_vy            , N * sizeof(real));
  cudaMalloc(k_vz            , N * sizeof(real));
  cudaMalloc(k_mass          , N * sizeof(real));
  cudaMalloc(k_kin_energy    , sizeof(real));
  cudaMalloc(k_type          , N * sizeof(int));
  if (wall) {
    cudaMalloc(k_walls_list    , Nlist_walls * sizeof(int));
    cudaMalloc(k_walls_start   , Nwalls * sizeof(int));
  }
  cudaMalloc(k_colloids_list , Nlist_colloids * sizeof(int));
  cudaMalloc(k_colloids_start, N_colloids * sizeof(int));  
  cudaMalloc(k_coll_x        , N_colloids * sizeof(real));
  cudaMalloc(k_coll_y        , N_colloids * sizeof(real));
  cudaMalloc(k_coll_z        , N_colloids * sizeof(real));
  cudaMalloc(k_coll_vx       , N_colloids * sizeof(real));
  cudaMalloc(k_coll_vy       , N_colloids * sizeof(real));
  cudaMalloc(k_coll_vz       , N_colloids * sizeof(real));
  cudaMalloc(k_coll_omegax   , N_colloids * sizeof(real));
  cudaMalloc(k_coll_omegay   , N_colloids * sizeof(real));
  cudaMalloc(k_coll_omegaz   , N_colloids * sizeof(real));
  cudaMalloc(k_coll_theta    , N_colloids * sizeof(real));
  cudaMalloc(k_coll_quat0    , N_colloids * sizeof(real));
  cudaMalloc(k_coll_quatx    , N_colloids * sizeof(real));
  cudaMalloc(k_coll_quaty    , N_colloids * sizeof(real));
  cudaMalloc(k_coll_quatz    , N_colloids * sizeof(real));        
  cudaMalloc(k_x_center      , Nlist_colloids * sizeof(real));
  cudaMalloc(k_y_center      , Nlist_colloids * sizeof(real));
  cudaMalloc(k_z_center      , Nlist_colloids * sizeof(real));
  cudaMalloc(k_time_subs     , N_subint * sizeof(real));
  //--- For the memory array, to avoid 2D arrays, a flattened array is built
  int size_memory = N * max_steps_memory;
  cudaMalloc(k_Bxx       , size_memory      * sizeof(real));
  cudaMalloc(k_Bxy       , size_memory      * sizeof(real));
  cudaMalloc(k_Byy       , size_memory      * sizeof(real));
  cudaMalloc(k_Bxz       , size_memory      * sizeof(real));
  cudaMalloc(k_Byz       , size_memory      * sizeof(real));
  cudaMalloc(k_Bzz       , size_memory      * sizeof(real));
  //--- mem_time stores the saved times ---
  cudaMalloc(k_mem_time  , max_steps_memory * sizeof(real));
  //-- k_time_subs_filter stores integer values (1 or 0) to filter the memory in the
  //   subintervals limits
  cudaMalloc(k_time_subs_filter, N_subint * sizeof(int));
  //-- head index, tail indices, Nfree_mem and free_mem are used to manage mem_time array -
  cudaMalloc(k_head_mem, sizeof(int));
  cudaMalloc(k_tail_mem  , max_steps_memory * sizeof(int));    
  cudaMalloc(k_Nfree_mem, sizeof(int));
  cudaMalloc(k_free_mem  , max_steps_memory * sizeof(int));
  cudaMalloc(k_posf, N * sizeof(float4));
  cudaMalloc(k_M_integration, max_steps_memory * sizeof(real));

  cudaMemcpy(*k_x,     x,     N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_y,     y,     N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_z,     z,     N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_vx,    vx,    N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_vy,    vy,    N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_vz,    vz,    N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_mass,  mass,  N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_kin_energy, &kin_energy,  sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_type,  type,  N * sizeof(int), cudaMemcpyHostToDevice);  
  if (wall) {
    cudaMemcpy(*k_walls_list, walls_list, Nlist_walls * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*k_walls_start, walls_start, Nwalls * sizeof(int), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(*k_colloids_list, colloids_list,
	     Nlist_colloids * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_colloids_start, colloids_start,
	     N_colloids * sizeof(int), cudaMemcpyHostToDevice);  
  cudaMemcpy(*k_coll_x,      coll_x     , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_y,      coll_y     , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_z,      coll_z     , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_vx,     coll_vx    , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_vy,     coll_vy    , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_vz,     coll_vz    , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_omegax, coll_omegax, N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_omegay, coll_omegay, N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_omegaz, coll_omegaz, N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_theta,  coll_theta , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_quat0,  coll_quat0 , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_quatx,  coll_quatx , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_quaty,  coll_quaty , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_coll_quatz,  coll_quatz , N_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_x_center,   x_center    , Nlist_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_y_center,   y_center    , Nlist_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_z_center,   z_center    , Nlist_colloids * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_time_subs,  time_subs   , N_subint * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Bxx,    Bxx,    size_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Bxy,    Bxy,    size_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Byy,    Byy,    size_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Bxz,    Bxz,    size_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Byz,    Byz,    size_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Bzz,    Bzz,    size_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_mem_time, mem_time, max_steps_memory * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_time_subs_filter, time_subs_filter, N_subint * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_head_mem, head_mem, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_tail_mem, tail_mem, max_steps_memory * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_Nfree_mem, Nfree_mem, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*k_free_mem, free_mem, max_steps_memory * sizeof(int), cudaMemcpyHostToDevice);  

  //--------- Other variables --------------
  cudaMalloc(k_fx             , N * sizeof(real));
  cudaMalloc(k_fy             , N * sizeof(real));
  cudaMalloc(k_fz             , N * sizeof(real));
  cudaMalloc(k_dens           , N * sizeof(real));
  cudaMalloc(k_inv_dens       , N * sizeof(real));  
  cudaMalloc(k_press          , N * sizeof(real));
  cudaMalloc(k_gradvxx        , N * sizeof(real));
  cudaMalloc(k_gradvxy        , N * sizeof(real));
  cudaMalloc(k_gradvxz        , N * sizeof(real));
  cudaMalloc(k_gradvyx        , N * sizeof(real));
  cudaMalloc(k_gradvyy        , N * sizeof(real));
  cudaMalloc(k_gradvyz        , N * sizeof(real));
  cudaMalloc(k_gradvzx        , N * sizeof(real));
  cudaMalloc(k_gradvzy        , N * sizeof(real));
  cudaMalloc(k_gradvzz        , N * sizeof(real));
  cudaMalloc(k_pressxx_over_dens_sq, N * sizeof(real));
  cudaMalloc(k_pressxy_over_dens_sq, N * sizeof(real));
  cudaMalloc(k_pressyy_over_dens_sq, N * sizeof(real));
  cudaMalloc(k_pressxz_over_dens_sq, N * sizeof(real));
  cudaMalloc(k_pressyz_over_dens_sq, N * sizeof(real));
  cudaMalloc(k_presszz_over_dens_sq, N * sizeof(real));          
  cudaMalloc(k_particle_cell  , N * sizeof(int));
  cudaMalloc(k_particle_index , N * sizeof(int));
  cudaMalloc(k_cell_start     , Ntotal_cells * sizeof(int));
  cudaMalloc(k_cell_end       , Ntotal_cells * sizeof(int));
  if (wall) {
    cudaMalloc(k_fx_wall        , Nwalls * sizeof(real));
    cudaMalloc(k_fy_wall        , Nwalls * sizeof(real));
    cudaMalloc(k_fz_wall        , Nwalls * sizeof(real));
  }
  cudaMalloc(k_fx_colloid     , N_colloids * sizeof(real));
  cudaMalloc(k_fy_colloid     , N_colloids * sizeof(real));
  cudaMalloc(k_fz_colloid     , N_colloids * sizeof(real));
  cudaMalloc(k_tx_colloid     , N_colloids * sizeof(real));
  cudaMalloc(k_ty_colloid     , N_colloids * sizeof(real));
  cudaMalloc(k_tz_colloid     , N_colloids * sizeof(real));
  cudaMalloc(k_coll_cell      , N_colloids * sizeof(int));
  cudaMalloc(k_coll_index     , N_colloids * sizeof(int));
  cudaMalloc(k_coll_cell_start, Ntotal_cells_colloids * sizeof(int));
  cudaMalloc(k_coll_cell_end  , Ntotal_cells_colloids * sizeof(int));
  //-- k_mem_Ntimes stores an integer: the number of saved times
  cudaMalloc(k_mem_Ntimes   , sizeof(int));  
  cudaMalloc(k_marked_memory_times, N_subint * sizeof(int));
  cudaMalloc(k_Nmarked, sizeof(int));
  cudaMalloc(k_dBxx_dt       , size_memory * sizeof(real));
  cudaMalloc(k_dBxy_dt       , size_memory * sizeof(real));
  cudaMalloc(k_dByy_dt       , size_memory * sizeof(real));
  cudaMalloc(k_dBxz_dt       , size_memory * sizeof(real));
  cudaMalloc(k_dByz_dt       , size_memory * sizeof(real));
  cudaMalloc(k_dBzz_dt       , size_memory * sizeof(real));
  cudaMalloc(k_stressxx_mem  , N * sizeof(real));
  cudaMalloc(k_stressxy_mem  , N * sizeof(real));
  cudaMalloc(k_stressyy_mem  , N * sizeof(real));
  cudaMalloc(k_stressxz_mem  , N * sizeof(real));
  cudaMalloc(k_stressyz_mem  , N * sizeof(real));
  cudaMalloc(k_stresszz_mem  , N * sizeof(real));  

  cudaMemset(*k_fx             , 0, N * sizeof(real));
  cudaMemset(*k_fy             , 0, N * sizeof(real));
  cudaMemset(*k_fz             , 0, N * sizeof(real));
  cudaMemset(*k_dens           , 0, N * sizeof(real));
  cudaMemset(*k_inv_dens       , 0, N * sizeof(real));                
  cudaMemset(*k_press          , 0, N * sizeof(real));
  cudaMemset(*k_gradvxx        , 0, N * sizeof(real));
  cudaMemset(*k_gradvxy        , 0, N * sizeof(real));
  cudaMemset(*k_gradvxz        , 0, N * sizeof(real));  
  cudaMemset(*k_gradvyx        , 0, N * sizeof(real));
  cudaMemset(*k_gradvyy        , 0, N * sizeof(real));
  cudaMemset(*k_gradvyz        , 0, N * sizeof(real));
  cudaMemset(*k_gradvzx        , 0, N * sizeof(real));
  cudaMemset(*k_gradvzy        , 0, N * sizeof(real));
  cudaMemset(*k_gradvzz        , 0, N * sizeof(real));
  cudaMemset(*k_pressxx_over_dens_sq, 0, N * sizeof(real));
  cudaMemset(*k_pressxy_over_dens_sq, 0, N * sizeof(real));
  cudaMemset(*k_pressyy_over_dens_sq, 0, N * sizeof(real));
  cudaMemset(*k_pressxz_over_dens_sq, 0, N * sizeof(real));
  cudaMemset(*k_pressyz_over_dens_sq, 0, N * sizeof(real));
  cudaMemset(*k_presszz_over_dens_sq, 0, N * sizeof(real));    
  cudaMemset(*k_dBxx_dt        , 0, size_memory * sizeof(real));
  cudaMemset(*k_dBxy_dt        , 0, size_memory * sizeof(real));
  cudaMemset(*k_dByy_dt        , 0, size_memory * sizeof(real));
  cudaMemset(*k_dBxz_dt        , 0, size_memory * sizeof(real));
  cudaMemset(*k_dByz_dt        , 0, size_memory * sizeof(real));
  cudaMemset(*k_dBzz_dt        , 0, size_memory * sizeof(real));
  cudaMemset(*k_stressxx_mem   , 0, N * sizeof(real));
  cudaMemset(*k_stressxy_mem   , 0, N * sizeof(real));
  cudaMemset(*k_stressyy_mem   , 0, N * sizeof(real));
  cudaMemset(*k_stressxz_mem   , 0, N * sizeof(real));
  cudaMemset(*k_stressyz_mem   , 0, N * sizeof(real));
  cudaMemset(*k_stresszz_mem   , 0, N * sizeof(real));  
  
  cudaMemset(*k_particle_cell  , 0, N * sizeof(int));              
  cudaMemset(*k_particle_index , 0, N * sizeof(int));
  cudaMemset(*k_cell_start     , 0, Ntotal_cells * sizeof(int));              
  cudaMemset(*k_cell_end       , 0, Ntotal_cells * sizeof(int));
  if (wall) {
    cudaMemset(*k_fx_wall        , 0, Nwalls * sizeof(real));
    cudaMemset(*k_fy_wall        , 0, Nwalls * sizeof(real));
    cudaMemset(*k_fz_wall        , 0, Nwalls * sizeof(real));
  }
  if (N_colloids > 0) {
    cudaMemset(*k_fx_colloid     , 0, N_colloids * sizeof(real));
    cudaMemset(*k_fy_colloid     , 0, N_colloids * sizeof(real));
    cudaMemset(*k_fz_colloid     , 0, N_colloids * sizeof(real));
    cudaMemset(*k_tx_colloid     , 0, N_colloids * sizeof(real));
    cudaMemset(*k_ty_colloid     , 0, N_colloids * sizeof(real));
    cudaMemset(*k_tz_colloid     , 0, N_colloids * sizeof(real));
    cudaMemset(*k_coll_cell      , 0, N_colloids * sizeof(int));              
    cudaMemset(*k_coll_index     , 0, N_colloids * sizeof(int));
    cudaMemset(*k_coll_cell_start, 0, Ntotal_cells_colloids * sizeof(int));              
    cudaMemset(*k_coll_cell_end  , 0, Ntotal_cells_colloids * sizeof(int));
  }
  //--- Some arras about memory of the integral model are initialized ---
  cudaMemset(*k_mem_Ntimes          , 0, sizeof(int));
  cudaMemset(*k_marked_memory_times , 0, N_subint * sizeof(int));
  cudaMemset(*k_Nmarked             , 0, sizeof(int));  

}
