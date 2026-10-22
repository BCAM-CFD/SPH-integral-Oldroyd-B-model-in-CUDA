/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"

#include <iostream>
#include <fstream>
#include <vector>

// Initialization of the computational particles when they are read from a file
int class_system::initialize_particles_old_sim(int N_subinter) {


  std::ifstream file;  // Declaration
  file.open("input_particles.in");  // Opening file
  
  if (!file.is_open()) {
    std::cerr << "Error al abrir el archivo." << std::endl;
    return 1;
  }

  //------ Variables to be read --------
  std::vector<real> x_r, y_r, z_r, vx_r, vy_r, vz_r, mass_r, type_r;
  real xi, yi, zi, vxi, vyi, vzi, massi, typei;

  //------ The file is read -------
  if (dim == 2)
    while (file >> xi >> yi >> vxi >> vyi >> massi >> typei) {
      x_r.push_back(xi);
      y_r.push_back(yi);
      vx_r.push_back(vxi);
      vy_r.push_back(vyi);
      mass_r.push_back(massi);
      type_r.push_back(typei);
    }
  else // dim == 3
    while (file >> xi >> yi >> zi >> vxi >> vyi >> vzi >> massi >> typei) {
      x_r.push_back(xi);
      y_r.push_back(yi);
      z_r.push_back(zi);      
      vx_r.push_back(vxi);
      vy_r.push_back(vyi);
      vz_r.push_back(vzi);      
      mass_r.push_back(massi);
      type_r.push_back(typei);
    }    

  file.close();

  //----- The vector is passed to the variables of the system object ------
  // N is defined
  this->N = x_r.size();
  // The arrays are constructed.
  x     = new real[N];
  y     = new real[N];
  z     = new real[N];
  vx    = new real[N];
  vy    = new real[N];
  vz    = new real[N];
  fx    = new real[N];
  fy    = new real[N];
  fz    = new real[N];  
  dens  = new real[N];
  press = new real[N];
  mass  = new real[N];
  type  = new int[N];

  time_subs           = new real[N_subinter];
  steps_memory_subint = new int[N_subinter];
  time_subs_filter    = new int[N_subinter];
  head_mem            = new int[1];
  Nfree_mem           = new int[1];
  stressxx_mem        = new real[N];
  stressxy_mem        = new real[N];
  stressyy_mem        = new real[N];
  stressxz_mem        = new real[N];
  stressyz_mem        = new real[N];
  stresszz_mem        = new real[N];      
  // The vectors are copied
  for (size_t i = 0; i < N; ++i) {
    x[i]    = x_r[i];
    y[i]    = y_r[i];
    vx[i]   = vx_r[i];
    vy[i]   = vy_r[i];
    mass[i] = mass_r[i];    
    type[i] = type_r[i];            
    if (dim == 3) {
      z[i]  = z_r[i];
      vz[i] = vz_r[i];
    }
  }

  return 0;

}
