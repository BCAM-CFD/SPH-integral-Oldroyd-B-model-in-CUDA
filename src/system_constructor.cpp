/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <math.h>
#include <stdio.h>

// Constructor of the class system
void class_system::constructor(int Nxyz[3],
			       int wall,
			       real overlap,
			       int dim,
			       int N_coll,
			       int new_sim,
			       int N_subinter) {

  this->dim = dim;

  //-- Walls_list and walls_start are constructed in system_initialize --  
  if (wall) {
    this->Nwalls = 2;
    fx_wall    = new real[this->Nwalls];
    fy_wall    = new real[this->Nwalls];
    fz_wall    = new real[this->Nwalls];
  }
  else
    this->Nwalls = 0;

  //-- Colloids_list and colloids_start are constructed in system_initialize --
  //   x_center, y_center and z_center are also constructed in system_initialize --
  if (N_coll > 0) {
    coll_x      = new real[N_coll];
    coll_y      = new real[N_coll];
    coll_z      = new real[N_coll];
    coll_vx     = new real[N_coll];
    coll_vy     = new real[N_coll];
    coll_vz     = new real[N_coll];
    coll_omegax = new real[N_coll];
    coll_omegay = new real[N_coll];
    coll_omegaz = new real[N_coll];
    coll_theta  = new real[N_coll];
    coll_quat0  = new real[N_coll];
    coll_quatx  = new real[N_coll];
    coll_quaty  = new real[N_coll];
    coll_quatz  = new real[N_coll];   
    fx_colloid  = new real[N_coll];
    fy_colloid  = new real[N_coll];
    fz_colloid  = new real[N_coll];
    tx_colloid  = new real[N_coll];
    ty_colloid  = new real[N_coll];
    tz_colloid  = new real[N_coll];
  }

  // If the particles are read from a file, these quantities are
  // already constructed at system_initialize_particles_old_sim
  if (new_sim == 1) { //--- New simulation ---
    //-- Walls are perpendicular to the y direction --
    int Ny;
    int extra_N_wall = 2*ceil(overlap);
    if (wall == 1) // With walls
      Ny = Nxyz[1] + 2*ceil(overlap);
    else // Without walls
      Ny = Nxyz[1]; 
    
    if (dim == 2) 
      N = Nxyz[0] * Ny;
    else   //-- dim = 3 --
      N = Nxyz[0] * Ny * Nxyz[2];

    //--- Just in case, some more memory is reserved ---
    if (N_coll > 0)
      N = (int)(N * 1.01);
    Nmax = N;

    x     = new real[Nmax];
    y     = new real[Nmax];
    z     = new real[Nmax];
    
    vx    = new real[Nmax];
    vy    = new real[Nmax];
    vz    = new real[Nmax];
    
    fx    = new real[Nmax];
    fy    = new real[Nmax];
    fz    = new real[Nmax];  
    
    dens  = new real[Nmax];
    
    press = new real[Nmax];
    
    mass  = new real[Nmax];
    
    type  = new int[Nmax];

    time_subs           = new real[N_subinter];
    steps_memory_subint = new int[N_subinter];
    time_subs_filter    = new int[N_subinter];
    head_mem            = new int[1];
    Nfree_mem           = new int[1];
    stressxx_mem        = new real[Nmax];
    stressxy_mem        = new real[Nmax];
    stressyy_mem        = new real[Nmax];
    stressxz_mem        = new real[Nmax];
    stressyz_mem        = new real[Nmax];
    stresszz_mem        = new real[Nmax];    
    //--- Finger tensor is allocated in system_initialize
    //--- mem_time is allocated in system_initialize
    //--- free_mem is allocated in system_initialize
    //--- tail_mem is allocated in system_initialize    
  }

}
