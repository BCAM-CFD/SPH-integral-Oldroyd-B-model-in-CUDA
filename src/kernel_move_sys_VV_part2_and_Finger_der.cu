/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//Part 2 of the function to move particles with a velocity Verlet method with lambda = 0.5
// and to calculate the time derivative of the Finger tensor
// The calculation of the derivative of the Finger tensor is
// dB/dt = (gradv)^T * B + B * gradv
// Expressed with tensorial indices
// dB/dt^{alpha beta} = gradv^{mu alpha} B^{mu beta} + B^{alpha mu} gradv^{mu beta}
__global__ void kernel_move_sys_VV_part2_and_Finger_der(real* __restrict__ vx,
							real* __restrict__ vy,
							real* __restrict__ vz,
							real* __restrict__ gradvxx,
							real* __restrict__ gradvxy,
							real* __restrict__ gradvxz,
							real* __restrict__ gradvyx,
							real* __restrict__ gradvyy,
							real* __restrict__ gradvyz,
							real* __restrict__ gradvzx,
							real* __restrict__ gradvzy,
							real* __restrict__ gradvzz,
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

  if (type[i] == 0) { //  If it is from fluid

    real half_dt_over_mass = 0.5 * dt / mass[i];    
    // Velocity at t + dt/2 
    vx[i] = vx[i] + half_dt_over_mass * fx[i];
    vy[i] = vy[i] + half_dt_over_mass * fy[i];
    if (dim == 3)
      vz[i] = vz[i] + half_dt_over_mass * fz[i];

    //---- Now the Finger tensor -----
    real half_dt = 0.5 * dt;
    int i_finger;    
    real gradvxx_i = gradvxx[i];
    real gradvxy_i = gradvxy[i];
    real gradvyx_i = gradvyx[i];
    real gradvyy_i = gradvyy[i];
    real gradvxz_i;
    real gradvyz_i;
    real gradvzx_i;
    real gradvzy_i;
    real gradvzz_i;
    if (dim == 3) {
      gradvxz_i = gradvxz[i];
      gradvyz_i = gradvyz[i];
      gradvzx_i = gradvzx[i];
      gradvzy_i = gradvzy[i];
      gradvzz_i = gradvzz[i];
    }

    //---- dB/dt tensors are calculated and stored, and Finger tensor
    //     at t + dt is also calculated ------
    // The memory position j of the particle i is in j * N + i
    if (dim == 2) {
      if (head_mem[0] >= 0) { //It is < 0 in the first step    
	int last_pos = head_mem[0];
	for (int j = 0; j < mem_Ntimes[0]; ++j) {
	  i_finger     = N * last_pos + i;
	  real Bxx_i = Bxx[i_finger];
	  real Bxy_i = Bxy[i_finger];
	  real Byy_i = Byy[i_finger];
          // dB/dt at t + dt	  
	  real dBxx_dt_i = 2.0 * (gradvxx_i * Bxx_i + gradvyx_i * Bxy_i);
	  real dBxy_dt_i = gradvyx_i * Byy_i + Bxx_i * gradvxy_i +
	                   Bxy_i * (gradvxx_i + gradvyy_i);
	  real dByy_dt_i = 2.0 * (gradvxy_i * Bxy_i + gradvyy_i * Byy_i);
          dBxx_dt[i_finger] = dBxx_dt_i;   // needed by VV_part1 in the next step
          dBxy_dt[i_finger] = dBxy_dt_i;
          dByy_dt[i_finger] = dByy_dt_i;
          // Finger tensor at t + dt
          Bxx[i_finger] = Bxx_i + half_dt * dBxx_dt_i;
          Bxy[i_finger] = Bxy_i + half_dt * dBxy_dt_i;
          Byy[i_finger] = Byy_i + half_dt * dByy_dt_i;	  
	  last_pos = tail_mem[last_pos];
	}
      }
    }
    else { // dim = 3
      if (head_mem[0] >= 0) { //It is < 0 in the first step    
	int last_pos = head_mem[0];
	for (int j = 0; j < mem_Ntimes[0]; ++j) {
	  i_finger     = N * last_pos + i;
	  real Bxx_i = Bxx[i_finger];
	  real Bxy_i = Bxy[i_finger];
	  real Byy_i = Byy[i_finger];
	  real Bxz_i = Bxz[i_finger];
	  real Byz_i = Byz[i_finger];
	  real Bzz_i = Bzz[i_finger];	  
          // dB/dt at t + dt
	real dBxx_dt_i = 2.0 * (gradvxx_i * Bxx_i + gradvyx_i * Bxy_i + gradvzx_i * Bxz_i);
	real dBxy_dt_i = gradvxx_i * Bxy_i + gradvyx_i * Byy_i + gradvzx_i * Byz_i + 
	  Bxx_i * gradvxy_i + Bxy_i * gradvyy_i + Bxz_i * gradvzy_i;
	real dByy_dt_i = 2.0 * (gradvxy_i * Bxy_i + gradvyy_i * Byy_i + gradvzy_i * Byz_i);
	real dBxz_dt_i = gradvxx_i * Bxz_i + gradvyx_i * Byz_i + gradvzx_i * Bzz_i +
	  Bxx_i * gradvxz_i + Bxy_i * gradvyz_i + Bxz_i * gradvzz_i;
	real dByz_dt_i = gradvxy_i * Bxz_i + gradvyy_i * Byz_i + gradvzy_i * Bzz_i +
	  Bxy_i * gradvxz_i + Byy_i * gradvyz_i + Byz_i * gradvzz_i;
	real dBzz_dt_i = 2.0 * (gradvxz_i * Bxz_i + gradvyz_i * Byz_i + gradvzz_i * Bzz_i);
	
          dBxx_dt[i_finger] = dBxx_dt_i;   // needed by VV_part1 in the next step
          dBxy_dt[i_finger] = dBxy_dt_i;
          dByy_dt[i_finger] = dByy_dt_i;
          dBxz_dt[i_finger] = dBxz_dt_i;
          dByz_dt[i_finger] = dByz_dt_i;
          dBzz_dt[i_finger] = dBzz_dt_i;	  
          // Finger tensor at t + dt
          Bxx[i_finger] = Bxx_i + half_dt * dBxx_dt_i;
          Bxy[i_finger] = Bxy_i + half_dt * dBxy_dt_i;
          Byy[i_finger] = Byy_i + half_dt * dByy_dt_i;
          Bxz[i_finger] = Bxz_i + half_dt * dBxz_dt_i;
          Byz[i_finger] = Byz_i + half_dt * dByz_dt_i;
          Bzz[i_finger] = Bzz_i + half_dt * dBzz_dt_i;	  	  	  
	  last_pos = tail_mem[last_pos];
	}
      }      
    
    }
  }

}

