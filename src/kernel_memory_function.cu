/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <math.h>

//-- Memory function of the Maxwell model --
__device__ real kernel_memory_function(real t,
				       real t_prime) {  
  real M = E/tau * exp(-(t-t_prime)/tau);
  return M;
}
