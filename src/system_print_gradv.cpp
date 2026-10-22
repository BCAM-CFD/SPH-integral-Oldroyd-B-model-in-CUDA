/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include "cuda_runtime.h"
#include "class_system.h"

// function to print system info
void class_system::print_gradv(real* k_x,
			       real* k_y,
			       real* k_z,
			       real* k_gradvxx,
			       real* k_gradvxy,
			       real* k_gradvxz,
			       real* k_gradvyx,
			       real* k_gradvyy,
			       real* k_gradvyz,
			       real* k_gradvzx,
			       real* k_gradvzy,
			       real* k_gradvzz,			       
			       int   step) {
  int Npart = this->N;

  char filename[50];
  sprintf(filename, "gradv-%d.dat", step);
  FILE* file = fopen(filename, "w");
  if (!file) {
    printf("System print gradv error: Error opening the file %s\n", filename);
    return;
  }

  // We use non-concordant names arrays to avoid the definition of more arrays
  cudaMemcpy(this->x,     k_x,       Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->y,     k_y,       Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->z,     k_z,       Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->vx,    k_gradvxx, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->vy,    k_gradvxy, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->vz,    k_gradvxz, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->mass,  k_gradvyx, Npart * sizeof(real), cudaMemcpyDeviceToHost);  
  cudaMemcpy(this->dens,  k_gradvyy, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->press, k_gradvyz, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->stressxx_mem, k_gradvzx, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->stressxy_mem, k_gradvzy, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->stressyy_mem, k_gradvzz, Npart * sizeof(real), cudaMemcpyDeviceToHost);
  if (dim == 2)
    for (int i = 0; i < Npart; ++i)
      fprintf(file, "%d " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " %d\n",
	      i,
	      this->x[i],
	      this->y[i],
	      this->vx[i],       // gradvxx[i]
	      this->vy[i],       // gradvxy[i]
	      this->mass[i],     // gradvyx[i]	      
	      this->dens[i]);    // gradvyy[i]
  else //--- dim == 3  ---
    for (int i = 0; i < Npart; ++i)
      fprintf(file, "%d " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " %d\n",
	      i,
	      this->x[i],
	      this->y[i],
	      this->z[i],
	      this->vx[i],            // gradvxx[i]
	      this->vy[i],            // gradvxy[i]
	      this->vz[i],            // gradvxz[i]
	      this->mass[i],          // gradvyx[i]	      
	      this->dens[i],          // gradvyy[i]
	      this->press[i],         // gradvyz[i]
	      this->stressxx_mem[i],  // gradvzx[i]
	      this->stressxy_mem[i],  // gradvzy[i]
	      this->stressyy_mem[i]); // gradvzz[i]
  fclose(file);

  printf("gradv file written. Time step %d\n", step);
}
