/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//Part 1 of the function to move particles with a velocity Verlet method with lambda = 0.5.
__global__ void kernel_move_sys_VV_part1(real* __restrict__ x,
					 real* __restrict__ y,
					 real* __restrict__ z,
					 real* __restrict__ vx,
					 real* __restrict__ vy,
					 real* __restrict__ vz,
					 real* __restrict__ Bxx,
					 real* __restrict__ Bxy,
					 real* __restrict__ Byy,
					 real* __restrict__ Bxz,
					 real* __restrict__ Byz,
					 real* __restrict__ Bzz,
					 real* __restrict__ fx,
					 real* __restrict__ fy,
					 real* __restrict__ fz,
					 real* __restrict__ dBxx_dt,
					 real* __restrict__ dBxy_dt,
					 real* __restrict__ dByy_dt,
					 real* __restrict__ dBxz_dt,
					 real* __restrict__ dByz_dt,
					 real* __restrict__ dBzz_dt,
					 real* __restrict__ mass,
					 int*  __restrict__ type,
					 int*  __restrict__ mem_Ntimes,
					 int*  __restrict__ head_mem,
					 int*  __restrict__ tail_mem) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) return;

  int type_i = type[i];
  if (type_i > 2) return; // particles of colloids are not moved here

  real xi  = x[i];
  real yi  = y[i];
  real zi;
  if (dim == 3)  
    zi  = z[i];

  if (type_i == 0) { // i fluid 
    real half_dt           = 0.5 * dt;        
    real half_dt_over_mass = 0.5 * dt / mass[i];
    // Velocity at t + dt/2 
    vx[i] = vx[i] + half_dt_over_mass * fx[i];
    vy[i] = vy[i] + half_dt_over_mass * fy[i];
    if (dim == 3)
      vz[i] = vz[i] + half_dt_over_mass * fz[i];

    // Finger tensor at t + dt/2
    int i_finger;
    if (dim == 2) {
      if (head_mem[0] >= 0) { //It is < 0 in the first step
    	int last_pos  = head_mem[0];
    	for (int j = 0; j < mem_Ntimes[0]; ++j) {
    	  i_finger      = N * last_pos + i;
	  Bxx[i_finger] = Bxx[i_finger] + half_dt * dBxx_dt[i_finger];
    	  Bxy[i_finger] = Bxy[i_finger] + half_dt * dBxy_dt[i_finger];
    	  Byy[i_finger] = Byy[i_finger] + half_dt * dByy_dt[i_finger];
    	  last_pos      = tail_mem[last_pos];
    	}
      }
    }
    else {  // dim = 3
      if (head_mem[0] >= 0) { //It is < 0 in the first step      
    	int last_pos  = head_mem[0];
    	for (int j = 0; j < mem_Ntimes[0]; ++j) {
    	  i_finger      = N * last_pos + i;
    	  Bxx[i_finger] = Bxx[i_finger] + half_dt * dBxx_dt[i_finger];
    	  Bxy[i_finger] = Bxy[i_finger] + half_dt * dBxy_dt[i_finger];
    	  Byy[i_finger] = Byy[i_finger] + half_dt * dByy_dt[i_finger];
    	  Bxz[i_finger] = Bxz[i_finger] + half_dt * dBxz_dt[i_finger];
    	  Byz[i_finger] = Byz[i_finger] + half_dt * dByz_dt[i_finger];
    	  Bzz[i_finger] = Bzz[i_finger] + half_dt * dBzz_dt[i_finger];      	
    	  last_pos      = tail_mem[last_pos];
    	}
      }
    }
  }
  else if (type_i == 1) {// i bottom wall
    vx[i] = V_bottom;
    vy[i] = 0.0;
    if (dim == 3)    
      vz[i] = 0.0;
  }
  else if (type_i == 2) {// i top wall
    vx[i] = V_top;
    vy[i] = 0.0;
    if (dim == 3)    
      vz[i] = 0.0;
  }  //Particles with type_i > 2 where discarded above

  // Position at t + dt
  xi = xi + vx[i] * dt;
  yi = yi + vy[i] * dt;
  if (dim == 3)
    zi = zi + vz[i] * dt;
  
  // Periodic Boundary conditions
  if (xi < 0)
    xi = xi + L[0];
  if (wall == 0) //--- If no wall ---
    if (yi < 0)
      yi = yi + L[1];
  if (dim == 3)
    if (zi < 0)
      zi = zi + L[2];
  
  if (xi > L[0])
    xi = xi - L[0];
  if (wall == 0) //--- If no wall ---    
    if (yi > L[1])
      yi = yi - L[1];
  if (dim == 3)
    if (zi > L[2])
      zi = zi - L[2];      

  // Data is stored  (v[i] is stored above, only when needed (when type_i == 0)  
  x[i] = xi;
  y[i] = yi;
  if (dim == 3)
    z[i] = zi;

}
