/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include "cuda_runtime.h"
#include "class_system.h"

// function to print the memory array of one given particle (i = particle)
void class_system::print_memory_array(int   particle,
				      real* k_Bxx,
				      real* k_Bxy,
				      real* k_Byy,
				      real* k_Bxz,
				      real* k_Byz,
				      real* k_Bzz,
				      real* k_mem_time,
				      int*  k_mem_Ntimes,
				      int*  k_head_mem,
				      int*  k_tail_mem,
				      int   step) {
  int Npart = this->N;

  char filename[50];
  sprintf(filename, "memory-%d.dat", step);
  FILE* file = fopen(filename, "w");
  if (!file) {
    printf("System print memory array error: Error opening the file %s\n", filename);
    return;
  }

  int size_memory     = N * max_steps_memory;  
  int mem_Ntimes;
  cudaMemcpy(this->head_mem, k_head_mem, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->tail_mem, k_tail_mem, max_steps_memory * sizeof(int), cudaMemcpyDeviceToHost);  
  cudaMemcpy(&mem_Ntimes, k_mem_Ntimes, sizeof(int), cudaMemcpyDeviceToHost);  
  cudaMemcpy(this->mem_time, k_mem_time, max_steps_memory * sizeof(real), cudaMemcpyDeviceToHost);  
  cudaMemcpy(this->Bxx,   k_Bxx,  size_memory * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->Bxy,   k_Bxy,  size_memory * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(this->Byy,   k_Byy,  size_memory * sizeof(real), cudaMemcpyDeviceToHost);
  if (dim == 3) {
    cudaMemcpy(this->Bxz, k_Bxz,  size_memory * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Byz, k_Byz,  size_memory * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Bzz, k_Bzz,  size_memory * sizeof(real), cudaMemcpyDeviceToHost);
  }

  // The memory array and the Finger tensor are written
  // The memory position j of the particle i is in i * max_steps_memory + j
  int i_finger;
  if (dim == 2) {
    int last_pos = head_mem[0];
    i_finger = N * last_pos + particle;   
    fprintf(file, "0 " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " \n",
	    mem_time[last_pos],
	    Bxx[i_finger],
	    Bxy[i_finger],
	    Byy[i_finger]);
    last_pos = tail_mem[last_pos];    
    for (int j = 1; j < mem_Ntimes; ++j) {
      i_finger = N * last_pos + particle;         
      fprintf(file, "%d " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " \n",
	      j,
	      mem_time[last_pos],
	      Bxx[i_finger],
	      Bxy[i_finger],
	      Byy[i_finger]);
      last_pos = tail_mem[last_pos];
    }
  }
  else {
    int last_pos = head_mem[0];
    i_finger = N * last_pos + particle;       
    fprintf(file, "0 " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " \n",
	    mem_time[last_pos],
	    Bxx[i_finger],
	    Bxy[i_finger],
	    Byy[i_finger],
	    Bxz[i_finger],
	    Byz[i_finger],
	    Bzz[i_finger]);
    last_pos = tail_mem[last_pos];        
    for (int j = 1; j < mem_Ntimes; ++j) {
      i_finger = N * last_pos + particle;               
      fprintf(file, "%d " REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT " \n",
	      j,
	      mem_time[last_pos],
	      Bxx[i_finger],
	      Bxy[i_finger],
	      Byy[i_finger],
	      Bxz[i_finger],
	      Byz[i_finger],
	      Bzz[i_finger]);
      last_pos = tail_mem[last_pos];
    }    
  }
  
  printf("Memory file of particle %d written. Time step %d\n", particle, step);
}
