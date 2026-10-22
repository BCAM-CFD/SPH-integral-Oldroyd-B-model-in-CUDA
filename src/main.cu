/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include <time.h>
#include "class_system.h"
#include "kernel_functions.h"
#include "config.h"

//---- Declaration of a function of main ------------
__host__ void print_main_vars(real delta_t,
			      int Nsteps,
			      int freq_micro,
			      int freq_macro,
			      int freq_walls,
			      int freq_colloids,
			      int new_sim,
			      int coll_move,
			      int N_coll);

//***************************
//   main program
//***************************
int main() {

  //-----------------------------------
  //----- Some declarations -----------
  //-----------------------------------  
  int error = 0;
  cudaError_t cuda_err;
  clock_t   start_neighbours;
  clock_t   end_neighbours;
  clock_t   start_forces;
  clock_t   end_forces;
  double    time_neighbours = 0.0;
  double    time_forces     = 0.0;

  //--------------------------------------------------------------------  
  //--- Declaration of the input variables (and their default values)---
  //--------------------------------------------------------------------    
  int  Nxyz[3]         = {20, 20, 20};
  real Lbox[3]         = {1.0, 1.0, 1.0};
  int  num_dim         = 2;
  real delta_t         = 0.01;
  int  Nsteps          = 1000;
  real overlap         = 4.0;
  real rho             = 1.0;
  real c               = 1.0;
  real P0              = 1.0;
  real eta             = 1.0;
  real zeta            = 0.0;
  real E_Maxwell       = 1.0;
  real relax_time      = 1.0;
  real t_cut           = 1.0;
  int  N_subinter      = 5;
  real x_integral      = 1.2;
  int  fr_filter       = 2;
  int  ext_f_type      = 0;  
  real ext_f[3]        = {0.0, 0.0, 0.0};
  int  freq_micro      = 1000;
  int  freq_macro      = 100;
  int  freq_walls      = 1000;
  int  freq_colloids   = 1000;    
  int  wall_host       = 0;
  real Vwall_top       = 0.0;
  real Vwall_bottom    = 0.0;
  int  N_coll          = 0;
  real coll_radius     = 1.0;
  real coll_density    = 1.0;
  real coll_x_data[MAX_COLLOIDS][MAX_VALUES_PER_COLLOID];
  int  coll_move       = 1;
  real coll_repulsion_cuton = 0.001;  
  int  new_sim         = 1;
  real F0_repulsion    = 1.0;
  real tau_repulsion   = 0.01;

  //--------------------------------------------------------------------  
  //---------- initialization ------------------------------------------
  //--------------------------------------------------------------------    
  // The system of particles is initialized in the host 
  class_system sys;

  //-------------------------------
  //-----  Input file is read  ----
  //-------------------------------  
  error = sys.read_input(Nxyz, Lbox, num_dim, delta_t, Nsteps,
			 overlap, rho, c, P0, eta, zeta,
			 ext_f_type, ext_f,
			 freq_micro, freq_macro, freq_walls, freq_colloids,
			 wall_host, Vwall_top, Vwall_bottom,
			 N_coll, coll_radius, coll_density, coll_x_data, coll_move,
			 coll_repulsion_cuton, F0_repulsion, tau_repulsion,
			 E_Maxwell, relax_time, t_cut, N_subinter, x_integral, fr_filter, new_sim);
  if (error != 0)
    return 1; // End of program

  //-------------------------------------------------------------------------
  //----- Main variables are printed (print_main_vars is defined below) -----
  //-------------------------------------------------------------------------  
  print_main_vars(delta_t, Nsteps, freq_micro, freq_macro, freq_walls,
		  freq_colloids, new_sim, coll_move, N_coll);

  //----------------------------------------------  
  //---- Initializing all pointers as nullptr ----
  //----------------------------------------------    
  sys.initialize_pointers();

  //------------------------------------  
  //----  The system is constructed ----
  //------------------------------------    
  sys.constructor(Nxyz, wall_host, overlap, num_dim, N_coll, new_sim, N_subinter);

  //------------------------------------    
  //---- The system is initialized -----
  //------------------------------------    
  error = sys.initialize(Nxyz, Lbox, num_dim, overlap, rho, c,
			 P0, eta, zeta, ext_f_type, ext_f,
			 wall_host, Vwall_top, Vwall_bottom,
			 N_coll, coll_radius, coll_density,
			 coll_x_data, coll_repulsion_cuton, F0_repulsion,
			 tau_repulsion, E_Maxwell, relax_time, t_cut, N_subinter,
			 x_integral, fr_filter, delta_t, new_sim);
  if (error != 0)
    return 1;  

  //----------------------------------  
  //---- System info is displayed ----
  //----------------------------------    
  sys.print_info();

  //---------------------------------------------------
  //---- Pointers in the device (GPU) are declared ----
  //---------------------------------------------------      
  real *k_x, *k_y, *k_z;
  real *k_vx, *k_vy, *k_vz;
  real *k_mass;
  real *k_fx, *k_fy, *k_fz;
  real *k_dens;
  real *k_inv_dens;  
  real *k_press;
  int  *k_particle_cell;
  int  *k_particle_index;
  int  *k_cell_start;
  int  *k_cell_end;
  real *k_kin_energy;
  int  *k_type;
  real *k_pressxx_over_dens_sq;
  real *k_pressxy_over_dens_sq;
  real *k_pressyy_over_dens_sq;
  real *k_pressxz_over_dens_sq;
  real *k_pressyz_over_dens_sq;
  real *k_presszz_over_dens_sq;  
  real *k_gradvxx, *k_gradvxy, *k_gradvxz;
  real *k_gradvyx, *k_gradvyy, *k_gradvyz;
  real *k_gradvzx, *k_gradvzy, *k_gradvzz;      
  real *k_fx_wall, *k_fy_wall, *k_fz_wall;
  int  *k_walls_list;
  int  *k_walls_start;
  int  *k_colloids_list;
  int  *k_colloids_start;  
  real *k_coll_x, *k_coll_y, *k_coll_z;
  real *k_coll_vx, *k_coll_vy, *k_coll_vz;
  real *k_coll_omegax, *k_coll_omegay, *k_coll_omegaz;
  real *k_coll_theta;
  real *k_coll_quat0, *k_coll_quatx, *k_coll_quaty, *k_coll_quatz;    
  real *k_x_center, *k_y_center, *k_z_center;
  real *k_fx_colloid, *k_fy_colloid, *k_fz_colloid;
  real *k_tx_colloid, *k_ty_colloid, *k_tz_colloid;
  int  *k_coll_cell;
  int  *k_coll_index;
  int  *k_coll_cell_start;
  int  *k_coll_cell_end;
  real *k_Bxx; // Finger tensors (a history of them)
  real *k_Bxy;
  real *k_Byy;
  real *k_Bxz;
  real *k_Byz;
  real *k_Bzz;
  real *k_dBxx_dt; // temporal derivative of Finger tensors (a history of them)
  real *k_dBxy_dt;
  real *k_dByy_dt;
  real *k_dBxz_dt;
  real *k_dByz_dt;
  real *k_dBzz_dt;  
  real *k_mem_time;            // Times stored in memory (integral model)
  real *k_time_subs;           // Limit times of subintervals (integral model)
  int  *k_time_subs_filter;    // Filter frequency between subintervals (integral model)
  int  *k_mem_Ntimes;          // Number of times stored in memory (integral model)
  int  *k_marked_memory_times; // Marked memory times to be deleted (integral model)
  int  *k_Nmarked;             // Number of marked memory times to be deleted (integral model)
  int  *k_head_mem;            // Head index, to manage times in memory (integral model)
  int  *k_tail_mem;            // Tail indices, to manage times in memory (integral model)
  int  *k_Nfree_mem;           // Number of free positions in k_mem_time array (integral model)
  int  *k_free_mem;            // Free positions in k_mem_time array (integral model)
  real *k_stressxx_mem;        // Stress of te memory term
  real *k_stressxy_mem;
  real *k_stressyy_mem;
  real *k_stressxz_mem;
  real *k_stressyz_mem;
  real *k_stresszz_mem;
  float4 *k_posf;              // position in float, to save time when checking neighbours
  real *k_M_integration;       // M values used to calculate stress via numerical integration

  //-------------------------------------------------------------      
  //----- Some variables (constants) are passed to the GPU ------
  //-------------------------------------------------------------        
  // The constants are declared at kernel_declare_constants.cu, and must be also 
  // specified in kernel_functions.h
  cudaMemcpyToSymbol(N                    , &sys.N                    , sizeof(int));
  cudaMemcpyToSymbol(dim                  , &sys.dim                  , sizeof(int));
  cudaMemcpyToSymbol(L                    , &sys.L                    , sizeof(real) * 3);
  cudaMemcpyToSymbol(cw                   , &sys.cw                   , sizeof(real));
  cudaMemcpyToSymbol(c_gradw              , &sys.c_gradw              , sizeof(real));
  cudaMemcpyToSymbol(rcut                 , &sys.rcut                 , sizeof(real));
  cudaMemcpyToSymbol(rcutsq               , &sys.rcutsq               , sizeof(real));
  cudaMemcpyToSymbol(rcut_inv             , &sys.rcut_inv             , sizeof(real));  
  cudaMemcpyToSymbol(csq                  , &sys.csq                  , sizeof(real));
  cudaMemcpyToSymbol(rho0                 , &sys.rho0                 , sizeof(real));
  cudaMemcpyToSymbol(a                    , &sys.a                    , sizeof(real));
  cudaMemcpyToSymbol(b                    , &sys.b                    , sizeof(real));
  cudaMemcpyToSymbol(dt                   , &delta_t                  , sizeof(real));
  cudaMemcpyToSymbol(cell_size            , &sys.cell_size            , sizeof(real) * 3);
  cudaMemcpyToSymbol(Ncells               , &sys.Ncells               , sizeof(int) * 3);
  cudaMemcpyToSymbol(Ntotal_cells         , &sys.Ntotal_cells         , sizeof(int));
  cudaMemcpyToSymbol(ext_force_type       , &sys.ext_force_type       , sizeof(int));    
  cudaMemcpyToSymbol(ext_force            , &sys.ext_force            , sizeof(real) * 3);
  cudaMemcpyToSymbol(ky                   , &sys.ky                   , sizeof(real));
  cudaMemcpyToSymbol(wall                 , &sys.wall                 , sizeof(int));
  cudaMemcpyToSymbol(y_bottom             , &sys.y_bottom             , sizeof(real));
  cudaMemcpyToSymbol(y_top                , &sys.y_top                , sizeof(real));
  cudaMemcpyToSymbol(V_bottom             , &sys.V_bottom             , sizeof(real));
  cudaMemcpyToSymbol(V_top                , &sys.V_top                , sizeof(real));
  cudaMemcpyToSymbol(Nwalls               , &sys.Nwalls               , sizeof(int));
  cudaMemcpyToSymbol(Nlist_walls          , &sys.Nlist_walls          , sizeof(int));
  cudaMemcpyToSymbol(N_colloids           , &sys.N_colloids           , sizeof(int));
  cudaMemcpyToSymbol(Nlist_colloids       , &sys.Nlist_colloids       , sizeof(int));  
  cudaMemcpyToSymbol(coll_R               , &sys.coll_R               , sizeof(real));
  cudaMemcpyToSymbol(coll_rho             , &sys.coll_rho             , sizeof(real));
  cudaMemcpyToSymbol(coll_mass            , &sys.coll_mass            , sizeof(real));
  cudaMemcpyToSymbol(coll_I               , &sys.coll_I               , sizeof(real));  
  cudaMemcpyToSymbol(beta_max             , &sys.beta_max             , sizeof(real));
  cudaMemcpyToSymbol(wall_width           , &sys.wall_width           , sizeof(real));
  cudaMemcpyToSymbol(coll_rep_cuton       , &sys.coll_rep_cuton       , sizeof(real));
  cudaMemcpyToSymbol(coll_rep_cutoff      , &sys.coll_rep_cutoff      , sizeof(real));  
  cudaMemcpyToSymbol(rcutoff_coll_rep     , &sys.rcutoff_coll_rep     , sizeof(real));
  cudaMemcpyToSymbol(rcuton_coll_rep      , &sys.rcuton_coll_rep      , sizeof(real));
  cudaMemcpyToSymbol(rcutoff_coll_rep_sq  , &sys.rcutoff_coll_rep_sq  , sizeof(real));
  cudaMemcpyToSymbol(rcuton_coll_rep_sq   , &sys.rcuton_coll_rep_sq   , sizeof(real));
  cudaMemcpyToSymbol(rcutoff_coll         , &sys.rcutoff_coll         , sizeof(real));
  cudaMemcpyToSymbol(rcutoff_coll_sq      , &sys.rcutoff_coll_sq      , sizeof(real));  
  cudaMemcpyToSymbol(cell_colloids_size   , &sys.cell_colloids_size   , sizeof(real) * 3);
  cudaMemcpyToSymbol(Ncells_colloids      , &sys.Ncells_colloids      , sizeof(int) * 3);
  cudaMemcpyToSymbol(Ntotal_cells_colloids, &sys.Ntotal_cells_colloids, sizeof(int));
  cudaMemcpyToSymbol(F0_rep               , &sys.F0_rep               , sizeof(real));
  cudaMemcpyToSymbol(tau_rep              , &sys.tau_rep              , sizeof(real));
  cudaMemcpyToSymbol(N_subint             , &sys.N_subint             , sizeof(int));
  cudaMemcpyToSymbol(freq_filter          , &sys.freq_filter          , sizeof(int));  
  cudaMemcpyToSymbol(max_steps_memory     , &sys.max_steps_memory     , sizeof(int));
  cudaMemcpyToSymbol(E                    , &sys.E                    , sizeof(real));
  cudaMemcpyToSymbol(tau                  , &sys.tau                  , sizeof(real));
  cudaMemcpyToSymbol(Lf                   , &sys.Lf                   , sizeof(float) * 3);
  cudaMemcpyToSymbol(rcutsq_f             , &sys.rcutsq_f             , sizeof(float));

  //--------------------------------------------------------  
  //---- Memory for pointers is allocated in the device ----
  //--------------------------------------------------------  
  sys.copy_pointers_to_device(&k_x, &k_y, &k_z, &k_vx, &k_vy, &k_vz,
			      &k_mass, &k_fx, &k_fy, &k_fz,
			      &k_dens, &k_inv_dens,
			      &k_press, &k_particle_cell, &k_particle_index,
			      &k_cell_start, &k_cell_end, &k_kin_energy, &k_type,
			      &k_pressxx_over_dens_sq, &k_pressxy_over_dens_sq,
			      &k_pressyy_over_dens_sq, &k_pressxz_over_dens_sq,
			      &k_pressyz_over_dens_sq, &k_presszz_over_dens_sq,
			      &k_gradvxx, &k_gradvxy, &k_gradvxz,
			      &k_gradvyx, &k_gradvyy, &k_gradvyz,
			      &k_gradvzx, &k_gradvzy, &k_gradvzz,
			      &k_fx_wall, &k_fy_wall, &k_fz_wall, &k_walls_list,
			      &k_walls_start, &k_colloids_list, &k_colloids_start,
			      &k_coll_x, &k_coll_y, &k_coll_z,
			      &k_coll_vx, &k_coll_vy, &k_coll_vz,
			      &k_coll_omegax, &k_coll_omegay, &k_coll_omegaz,
			      &k_coll_theta, &k_coll_quat0, &k_coll_quatx,
			      &k_coll_quaty, &k_coll_quatz,
			      &k_x_center, &k_y_center, &k_z_center,
			      &k_fx_colloid, &k_fy_colloid, &k_fz_colloid,	
			      &k_tx_colloid, &k_ty_colloid, &k_tz_colloid,
			      &k_coll_cell, &k_coll_index,
			      &k_coll_cell_start, &k_coll_cell_end,
			      &k_Bxx, &k_Bxy, &k_Byy, &k_Bxz, &k_Byz, &k_Bzz,
			      &k_dBxx_dt, &k_dBxy_dt, &k_dByy_dt,
			      &k_dBxz_dt, &k_dByz_dt, &k_dBzz_dt,
			      &k_mem_time, &k_time_subs,
			      &k_time_subs_filter, &k_mem_Ntimes,
			      &k_marked_memory_times, &k_Nmarked,
			      &k_head_mem, &k_tail_mem, &k_Nfree_mem, &k_free_mem,
			      &k_stressxx_mem, &k_stressxy_mem, &k_stressyy_mem,
			      &k_stressxz_mem, &k_stressyz_mem, &k_stresszz_mem,
			      &k_posf, &k_M_integration);

  //------------------------------------  
  //--- Configure blocks and threads ---
  //------------------------------------
  //--- Changes in this value can modified performance of the code ---
  //***** If there are a small number of particles, 32 or 64 is faster. For a large
  //       number of particles, 128 is faster *****/
  dim3 threadsPerBlock(32);
  dim3 numBlocks((sys.N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  //-------------------------------------------------    
  //--- Constants stored in the GPU are displayed ---
  //-------------------------------------------------      
  kernel_print_constants<<<1, 1>>>(k_time_subs);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf("Error en kernel_print_constants: %s\n", cudaGetErrorString(cuda_err));
    return 1;
  }

  //--------------------------------  
  //---- First Neighbour search ----
  //--------------------------------    
  error = sys.neighbours_search(numBlocks, threadsPerBlock, k_x, k_y, k_z,
				k_particle_cell, k_particle_index,
				k_cell_start, k_cell_end,				  
				k_coll_x, k_coll_y, k_coll_z,
				k_coll_cell, k_coll_index,
				k_coll_cell_start, k_coll_cell_end);
  if (error != 0)
    return 1; // End of program  

  //-----------------------------------------------------------  
  //--- Particle forces are calculated in the initial state ---
  //-----------------------------------------------------------    
  //--- (needed for the Velocity Verlet algorithm) ---
  //--- Density and pressure are also calculated in this function
  error = sys.calculate_forces(numBlocks, threadsPerBlock, k_x, k_y, k_z,
			       k_vx, k_vy, k_vz,
			       k_fx, k_fy, k_fz, 
			       k_gradvxx, k_gradvxy, k_gradvxz, k_gradvyx, k_gradvyy,
			       k_gradvyz, k_gradvzx, k_gradvzy, k_gradvzz,
			       k_Bxx, k_Bxy, k_Byy, k_Bxz, k_Byz, k_Bzz,
			       k_dBxx_dt, k_dBxy_dt, k_dByy_dt,
			       k_dBxz_dt, k_dByz_dt, k_dBzz_dt,
			       k_mem_Ntimes, k_mem_time, k_head_mem, k_tail_mem,
			       k_press, k_stressxx_mem, k_stressxy_mem, k_stressyy_mem,
			       k_stressxz_mem, k_stressyz_mem, k_stresszz_mem,
			       k_pressxx_over_dens_sq, k_pressxy_over_dens_sq,
			       k_pressyy_over_dens_sq, k_pressxz_over_dens_sq,
			       k_pressyz_over_dens_sq, k_presszz_over_dens_sq,
			       k_dens, k_inv_dens, k_mass, k_particle_index,
			       k_cell_start, k_cell_end, k_type,
			       k_fx_wall, k_fy_wall, k_fz_wall,
			       k_coll_x, k_coll_y, k_coll_z,
			       k_coll_vx, k_coll_vy, k_coll_vz,
			       k_coll_omegax, k_coll_omegay, k_coll_omegaz,
			       k_fx_colloid, k_fy_colloid, k_fz_colloid,
			       k_tx_colloid, k_ty_colloid, k_tz_colloid,
			       k_colloids_list, k_colloids_start, k_coll_index,
			       k_coll_cell_start, k_coll_cell_end, k_posf,
			       k_M_integration);
  if (error != 0)
    return 1;

  //--------------------------------  
  //--- Initial state is printed ---
  //--------------------------------    
  sys.print_output(numBlocks, threadsPerBlock,
		   k_x, k_y, k_z, k_vx, k_vy, k_vz, k_fx, k_fy, k_fz,
		   k_mass, k_type, k_dens, k_press,
		   k_stressxx_mem, k_stressxy_mem, k_stressyy_mem,
		   k_stressxz_mem, k_stressyz_mem, k_stresszz_mem,		     
		   k_fx_wall, k_fy_wall, k_fz_wall,k_walls_list, k_walls_start, 
		   k_coll_x, k_coll_y, k_coll_z,
		   k_coll_vx, k_coll_vy, k_coll_vz,
		   k_coll_omegax, k_coll_omegay, k_coll_omegaz,			 
		   k_fx_colloid, k_fy_colloid, k_fz_colloid,
		   k_tx_colloid, k_ty_colloid, k_tz_colloid,
		   k_kin_energy, freq_micro, freq_walls, freq_colloids,
		   freq_macro, 0);


  //-------------------------------------------------------------------------------------- 
  //----------------------------------- Main loop ----------------------------------------
  //--------------------------------------------------------------------------------------
  //
  clock_t start = clock();
  for (int step = 1; step <= Nsteps; ++step) {
    
    //---- Particles are moved (part1 of the Velocity Verlet algorithm) ---
    error = sys.V_Verlet_step1(numBlocks, threadsPerBlock,
			       k_x, k_y, k_z, k_vx, k_vy, k_vz,
			       k_Bxx, k_Bxy, k_Byy, k_Bxz, k_Byz, k_Bzz,
			       k_fx, k_fy, k_fz,
			       k_dBxx_dt, k_dBxy_dt, k_dByy_dt,
			       k_dBxz_dt, k_dByz_dt, k_dBzz_dt,
			       k_mass, k_type, k_mem_Ntimes,
			       k_head_mem, k_tail_mem,
			       k_coll_x, k_coll_y, k_coll_z,
			       k_coll_vx, k_coll_vy, k_coll_vz,
			       k_coll_omegax, k_coll_omegay, k_coll_omegaz,
			       k_coll_theta, k_coll_quat0, k_coll_quatx,
			       k_coll_quaty, k_coll_quatz,
			       k_fx_colloid, k_fy_colloid, k_fz_colloid,
			       k_tx_colloid, k_ty_colloid, k_tz_colloid,
			       k_x_center, k_y_center, k_z_center,
			       k_colloids_list, k_colloids_start, coll_move);
    if (error != 0)
      return 1; // End of program

    //---- Neighbour search ----
    start_neighbours = clock();    
    error = sys.neighbours_search(numBlocks, threadsPerBlock, k_x, k_y, k_z,
				  k_particle_cell, k_particle_index,
				  k_cell_start, k_cell_end,				  
				  k_coll_x, k_coll_y, k_coll_z,
				  k_coll_cell, k_coll_index,
				  k_coll_cell_start, k_coll_cell_end);
    end_neighbours = clock();
    time_neighbours = time_neighbours +
      (double)(end_neighbours - start_neighbours) / CLOCKS_PER_SEC;    
    if (error != 0)
      return 1; // End of program

    //----- Forces are calculated -----
    start_forces = clock();            
    error = sys.calculate_forces(numBlocks, threadsPerBlock, k_x, k_y, k_z,
				 k_vx, k_vy, k_vz, k_fx, k_fy, k_fz,
				 k_gradvxx, k_gradvxy, k_gradvxz, k_gradvyx, k_gradvyy,
				 k_gradvyz, k_gradvzx, k_gradvzy, k_gradvzz,
				 k_Bxx, k_Bxy, k_Byy, k_Bxz, k_Byz, k_Bzz,
				 k_dBxx_dt, k_dBxy_dt, k_dByy_dt,
				 k_dBxz_dt, k_dByz_dt, k_dBzz_dt,
				 k_mem_Ntimes, k_mem_time, k_head_mem, k_tail_mem,
				 k_press, k_stressxx_mem, k_stressxy_mem, k_stressyy_mem,
				 k_stressxz_mem, k_stressyz_mem, k_stresszz_mem,
				 k_pressxx_over_dens_sq, k_pressxy_over_dens_sq,
				 k_pressyy_over_dens_sq, k_pressxz_over_dens_sq,
				 k_pressyz_over_dens_sq, k_presszz_over_dens_sq,
				 k_dens, k_inv_dens, k_mass, k_particle_index,
				 k_cell_start, k_cell_end, k_type,
				 k_fx_wall, k_fy_wall, k_fz_wall,
				 k_coll_x, k_coll_y, k_coll_z,
				 k_coll_vx, k_coll_vy, k_coll_vz,
				 k_coll_omegax, k_coll_omegay, k_coll_omegaz,
				 k_fx_colloid, k_fy_colloid, k_fz_colloid,
				 k_tx_colloid, k_ty_colloid, k_tz_colloid,
				 k_colloids_list, k_colloids_start, k_coll_index,
				 k_coll_cell_start, k_coll_cell_end, k_posf,
				 k_M_integration);
    end_forces = clock();
    time_forces = time_forces + (double)(end_forces - start_forces) / CLOCKS_PER_SEC;  
    if (error != 0)
      return 1; // End of program                
    
    //---- Particles are moved (part2 of the Velocity Verlet algorithm) ----
    error = sys.V_Verlet_step2_and_Finger_der(numBlocks, threadsPerBlock,
					      k_vx, k_vy, k_vz,
					      k_gradvxx, k_gradvxy, k_gradvxz,
					      k_gradvyx, k_gradvyy, k_gradvyz,
					      k_gradvzx, k_gradvzy, k_gradvzz,
					      k_Bxx, k_Bxy, k_Byy, k_Bxz, k_Byz, k_Bzz,
					      k_fx, k_fy, k_fz,
					      k_dBxx_dt, k_dBxy_dt, k_dByy_dt,
					      k_dBxz_dt, k_dByz_dt, k_dBzz_dt,
					      k_mass, k_type,
					      k_mem_Ntimes, k_head_mem, k_tail_mem,
					      k_coll_vx, k_coll_vy, k_coll_vz,
					      k_coll_omegax, k_coll_omegay, k_coll_omegaz,
					      k_fx_colloid, k_fy_colloid, k_fz_colloid,
					      k_tx_colloid, k_ty_colloid, k_tz_colloid,
					      coll_move);
    if (error != 0)
      return 1; // End of program

    //---- The memory array is updated ----
    error = sys.update_memory(numBlocks, threadsPerBlock,
			      k_Bxx, k_Bxy, k_Byy, k_Bxz, k_Byz, k_Bzz,
			      k_mem_time, k_mem_Ntimes, k_time_subs, k_time_subs_filter,
			      k_marked_memory_times, k_Nmarked,
			      k_head_mem, k_tail_mem, k_Nfree_mem, k_free_mem, k_type);

    //--- Output is written in output files ---
    sys.print_output(numBlocks, threadsPerBlock,
		     k_x, k_y, k_z, k_vx, k_vy, k_vz, k_fx, k_fy, k_fz,
		     k_mass, k_type, k_dens, k_press,
		     k_stressxx_mem, k_stressxy_mem, k_stressyy_mem,
		     k_stressxz_mem, k_stressyz_mem, k_stresszz_mem,		     
		     k_fx_wall, k_fy_wall, k_fz_wall,  k_walls_list, k_walls_start, 
		     k_coll_x, k_coll_y, k_coll_z,
		     k_coll_vx, k_coll_vy, k_coll_vz,
		     k_coll_omegax, k_coll_omegay, k_coll_omegaz,			 
		     k_fx_colloid, k_fy_colloid, k_fz_colloid,
		     k_tx_colloid, k_ty_colloid, k_tz_colloid,
		     k_kin_energy, freq_micro, freq_walls, freq_colloids,
		     freq_macro, step); 
  }  //--------- End of main loop ------------

  //---------------------------------------
  //---- Computation time is displayed ----
  //---------------------------------------  
  clock_t end = clock();
  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Total time: %f seconds\n", time_spent);
  printf("Neighbours time: %f seconds\n", time_neighbours);
  printf("Forces time: %f seconds\n", time_forces);  

  //---------------------------------
  //---- Resources are released -----
  //---------------------------------  
  cudaFree(k_x);
  cudaFree(k_y);
  cudaFree(k_z);
  cudaFree(k_vx);
  cudaFree(k_vy);
  cudaFree(k_vz);
  cudaFree(k_mass);
  cudaFree(k_fx);
  cudaFree(k_fy);
  cudaFree(k_fz);
  cudaFree(k_gradvxx);
  cudaFree(k_gradvxy);
  cudaFree(k_gradvxz);
  cudaFree(k_gradvyx);
  cudaFree(k_gradvyy);
  cudaFree(k_gradvyz);
  cudaFree(k_gradvzx);
  cudaFree(k_gradvzy);
  cudaFree(k_gradvzz);          
  cudaFree(k_dens);
  cudaFree(k_inv_dens);  
  cudaFree(k_press);
  cudaFree(k_particle_cell);
  cudaFree(k_particle_index);
  cudaFree(k_cell_start);
  cudaFree(k_cell_end);
  cudaFree(k_type);
  cudaFree(k_Bxx);
  cudaFree(k_Bxy);
  cudaFree(k_Byy);
  cudaFree(k_Bxz);
  cudaFree(k_Byz);
  cudaFree(k_Bzz);
  cudaFree(k_dBxx_dt);
  cudaFree(k_dBxy_dt);
  cudaFree(k_dByy_dt);
  cudaFree(k_dBxz_dt);
  cudaFree(k_dByz_dt);
  cudaFree(k_dBzz_dt);    
  cudaFree(k_mem_time);
  cudaFree(k_time_subs);
  cudaFree(k_time_subs_filter);
  cudaFree(k_mem_Ntimes);
  cudaFree(k_marked_memory_times);
  cudaFree(k_Nmarked);
  cudaFree(k_head_mem);
  cudaFree(k_tail_mem);    
  cudaFree(k_Nfree_mem);
  cudaFree(k_free_mem);
  cudaFree(k_stressxx_mem);
  cudaFree(k_stressxy_mem);
  cudaFree(k_stressyy_mem);
  cudaFree(k_stressxz_mem);
  cudaFree(k_stressyz_mem);
  cudaFree(k_stresszz_mem);
  cudaFree(k_pressxx_over_dens_sq);
  cudaFree(k_pressxy_over_dens_sq);
  cudaFree(k_pressyy_over_dens_sq);
  cudaFree(k_pressxz_over_dens_sq);
  cudaFree(k_pressyz_over_dens_sq);
  cudaFree(k_presszz_over_dens_sq);
  cudaFree(k_posf);
  cudaFree(k_M_integration);
  if (sys.wall) {
    cudaFree(k_fx_wall);
    cudaFree(k_fy_wall);
    cudaFree(k_fz_wall);
    cudaFree(k_walls_list);
    cudaFree(k_walls_start);
  }
  if (sys.N_colloids > 0) {
    cudaFree(k_colloids_list);
    cudaFree(k_coll_x);
    cudaFree(k_coll_y);
    cudaFree(k_coll_z);
    cudaFree(k_coll_vx);
    cudaFree(k_coll_vy);
    cudaFree(k_coll_vz);
    cudaFree(k_coll_omegax);
    cudaFree(k_coll_omegay);
    cudaFree(k_coll_omegaz);
    cudaFree(k_coll_theta);    
    cudaFree(k_x_center);
    cudaFree(k_y_center);
    cudaFree(k_z_center);
    cudaFree(k_fx_colloid);
    cudaFree(k_fy_colloid);
    cudaFree(k_fz_colloid);
    cudaFree(k_tx_colloid);
    cudaFree(k_ty_colloid);
    cudaFree(k_tz_colloid);
    cudaFree(k_coll_cell);
    cudaFree(k_coll_index);
    cudaFree(k_coll_cell_start);
    cudaFree(k_coll_cell_end);
  }
  cudaDeviceReset();      

  //-------------------------------------  
  //---- The sys object is destroyed ----
  //-------------------------------------    
  sys.destructor(); 
  
  return 0;
}  //-------------- End of main ------------------





//-----------------------------------------------
//----  Function  to print variables of main ----
__host__ void print_main_vars(real delta_t,
		     int  Nsteps,
		     int  freq_micro,
		     int  freq_macro,
		     int  freq_walls,
		     int  freq_colloids,
		     int  new_sim,
		     int  coll_move,
		     int  N_coll) {

  std::cout << "--- Main variables ----\n";
  std::cout << "Time step                 = " << delta_t  << "\n";
  std::cout << "Number of steps           = " << Nsteps  << "\n";
  std::cout << "Output frequency micro    = " << freq_micro  << "\n";
  std::cout << "Output frequency macro    = " << freq_macro  << "\n";
  std::cout << "Output frequency walls    = " << freq_walls  << "\n";
  std::cout << "Output frequency colloids = " << freq_colloids  << "\n";
  if (new_sim == 1)
    std::cout << "New simulation?           = Yes\n";
  else
    std::cout << "New simulation?           = No\n";    
  if (N_coll > 0)
    if (coll_move == 1) 
      std::cout << "Are colloids moving?      = Yes\n";
    else
      std::cout << "Are colloids moving?      = No\n";
  std::cout << "coll_move                 = " << coll_move << "\n";
  std::cout << "-----------------------\n";  

}
		     
