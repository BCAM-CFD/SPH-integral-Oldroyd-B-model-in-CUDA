# SPH integral Oldroyd-B model in CUDA

SPH for a viscoelastic fluid described by an **integral constitutive model**,
with wall boundaries and suspensions of spherical particles, implemented in CUDA.

The polymeric stress is computed from the history of the Finger tensor **B**(t, t')
of every fluid particle:

    sigma_p(t) = - int_{t - t_cut}^{t} M(t - t') [ I - B(t, t') ] dt'

with the Maxwell memory function

    M(s) = (E / tau) exp(-s / tau)

(i.e. the Lodge / upper-convected Maxwell model in integral form; together with the
Newtonian viscosities `eta` and `zeta` of the solvent it corresponds to an Oldroyd-B fluid).
The integral is evaluated with the trapezoidal rule over the stored memory times.

Developed by Adolfo Vázquez-Quesada.

mail: a.vazquez-quesada@fisfun.uned.es

--------------------------------------------------

## Structure of the code

- Host variables and functions are declared in class_system.h.
  Each function is defined in its own file, specifically in system_* files.

- Device variables and functions are declared in kernel_functions.h.
  Each function is defined in its own file, specifically in kernel_* files.
  Device constants are declared in kernel_declare_constants.cu (and must also be
  declared in kernel_functions.h).

- In config.h we can define if the real variables are float or double.
  It also defines MAX_COLLOIDS (maximum number of colloids that can be read
  from the input file, default 1000).

- atomicAdd_double.cuh provides atomicAdd for double precision.

- main.cu is the main file of the code.

### Memory (history) of the integral model

The memory window [t - t_cut, t] is divided into `N_subinter` subintervals whose
lengths grow geometrically with ratio `x_integral` (the most recent subinterval is
the shortest). In subinterval i (i = 0 is the most recent one) the Finger tensor is
stored every `dt * freq_filter^i`, so old history is kept with a coarser time
resolution. The stored times are managed as a linked list (head/tail indices and a
list of free positions), and times older than `t_cut` are discarded.

The memory needed in the GPU is proportional to N × max_steps_memory, where
max_steps_memory is the total number of stored times (printed at the beginning of
the run). Large `t_cut`, small `dt`, small `x_integral` or small `freq_filter`
increase it.

## Compilation 

The compilation is done with the makefile file. 

**Before compiling**, check the flag `-arch=sm_XX` in the makefile: it must match the
compute capability of your GPU (e.g. sm_86 for Ampere, sm_75 for Turing, sm_61 for Pascal, etc).

To compile, just write the following command in the command line:

make

If you want to clean everything before compiling again, just do

make clean

## Running a simulation

You need to have the following files in the same directory (with the exact names shown below):

- **SPH_program**: the executable.
- **input**: a file containing the program inputs.
- **input_particles.in**: only when restarting a simulation (`new_sim 0`, see below).

To run the program, execute:

./SPH_program

## Input variables 

Each line of the input file has the form `name value(s)`. Lines whose first word is
not a known variable (for instance, lines starting with `#`) are ignored, so they can
be used as comments. **All variables listed below are compulsory** (except `coll_x`
when `N_colloids 0`); the program stops if one of them is missing.
Vector variables (`N`, `L`, `ext_force`, `coll_x`) must have `dim` components.

**N**       -> Number of particles in each direction (in a rectangular grid)

**L**       -> Box length in each direction.

**dim**     -> Number of dimensions (2 or 3)

**dt**      -> Time step

**Nsteps**  -> Number of steps

**overlap** -> rcut / dx   (where dx = L/N)

**rho**     -> fluid density

**c**       -> Speed of sound

**P0**      -> parameter of the equation of state

**eta**     -> shear viscosity (Newtonian solvent)

**zeta**    -> bulk viscosity (Newtonian solvent)

**E**       -> Elastic modulus of the Maxwell memory function M(s) = (E/tau) exp(-s/tau)

**tau**     -> Relaxation time of the memory function

**t_cut**   -> Length of the memory window. The history older than t_cut is neglected
               (it should be several times tau).

**N_subinter** -> Number of subintervals in which the memory window is divided.

**x_integral** -> Ratio between the lengths of consecutive subintervals (must be > 1).

**freq_filter** -> Filter factor between consecutive subintervals: in subinterval i the
                   Finger tensor is stored every dt * freq_filter^i (must be > 1).

**ext_force_type** -> Type of external force (applied only on fluid particles):

	       0: No external force

		   1: Constant force

		   2: sine force: F_x = m * ext_force_x * sin(2 pi y / L_y)

		   3: force to simulate inverse Poiseuille flow
		      (+ext_force_x for y < L_y/2, -ext_force_x for y > L_y/2)

**ext_force** -> External force vector (per unit mass). It must appear in the input file
                 even if ext_force_type = 0.

**freq_micro**    -> Frequency to write micro files.

**freq_macro**    -> Frequency to write macro file.

**freq_walls**    -> Frequency to write walls file.

**freq_colloids** -> Frequency to write colloids files.

**wall** -> walls are normal to the y-direction.

		0: no walls

     	1: walls

**Vwall_bottom** -> x-velocity of the bottom wall.

**Vwall_top**    -> x-velocity of the top wall.

**N_colloids** -> Number of colloidal particles.

**coll_R**     -> Colloids radius.

**coll_rho**   -> Colloids density.

**coll_move**  -> whether colloids are moving or fixed

			  0: colloids are fixed

	   		  1: colloids can move.

**coll_x**     -> Position of one colloid. Put as many coll_x lines as N_colloids.

**coll_repulsion_cuton** -> Cut on of the repulsion force between colloids
                            (gap below which the repulsion force is kept constant).

**F0_repulsion**         -> Magnitude of the repulsion force.

**tau_repulsion**        -> Tau parameter of the repulsion force. The repulsion force between
                            two colloids at distance r is

                            F = F0 * tau * exp(-tau*(r-2R)/R) / (1 - exp(-tau*(r-2R)/R))

                            and its cut off (gap) is 5 R / tau_repulsion.

**new_sim** -> The simulation is new, or it is restarting from a previous simulation

		   0: existing simulation (particles are read from input_particles.in)

	       1: new simulation

## Key of the output files

Type of particle: 0 -> fluid, 1 -> bottom wall, 2 -> top wall, 3 + k -> colloid k.

stress_mem is the polymeric (memory) stress computed with the integral model
(it is only calculated for fluid particles).

micro-\<step\>.dat (one file every freq_micro steps): 

		-- 2D --

       1  -> Particle id

       2  -> x

       3  -> y

       4  -> vel x

       5  -> vel y

       6  -> mass

       7  -> density

       8  -> pressure

       9  -> stress_mem xx

       10 -> stress_mem xy

       11 -> stress_mem yy

       12 -> type of particle

       -- 3D --

       1  -> Particle id

       2  -> x

       3  -> y

       4  -> z   

       5  -> vel x

       6  -> vel y

       7  -> vel z    

       8  -> mass

       9  -> density

       10 -> pressure

       11 -> stress_mem xx

       12 -> stress_mem xy

       13 -> stress_mem yy

       14 -> stress_mem xz

       15 -> stress_mem yz

       16 -> stress_mem zz

       17 -> type of particle

walls.dat (one line every freq_walls steps; only if wall = 1): 

		-- 2D --

       1 -> Step

       2 -> force x on bottom wall

       3 -> force y on bottom wall

       4 -> force x on top wall

       5 -> force y on top wall

       -- 3D --

       1 -> Step

       2 -> force x on bottom wall

       3 -> force y on bottom wall

       4 -> force z on bottom wall

	   5 -> force x on top wall

       6 -> force y on top wall

       7 -> force z on top wall


colloids-\<step\>.dat (one file every freq_colloids steps; only if N_colloids > 0): 

	  -- 2D --

	  1 -> Colloid id

	  2 -> x

	  3 -> y

	  4 -> vel x

	  5 -> vel y

	  6 -> ang vel

	  7 -> force x

	  8 -> force y

	  9 -> torque

	  -- 3D --

	  1  -> Colloid id

	  2  -> x

	  3  -> y

	  4  -> z	 

	  5  -> vel x

	  6  -> vel y

	  7  -> vel z	

	  8  -> ang vel x

	  9  -> ang vel y

	  10 -> ang vel z	

	  11 -> force x

	  12 -> force y

	  13 -> force z	  

	  14 -> torque x

	  15 -> torque y

	  16 -> torque z

macro.dat (one line every freq_macro steps): 

	   1 -> Step

       2 -> Total kinetic energy

## To restart a simulation

**Important:** the deformation history (Finger tensors stored in memory) is not saved
in the output files and is not read when restarting. The restarted simulation begins
with an empty memory (B = I), i.e. as if the fluid were fully relaxed, so the
polymeric stress needs a transient of the order of tau to recover. This is something to
be implemented.

1.- Use the same input file of the initial simulation (or some variation of it).

2.- In the input file: new_sim 1

3.- If your simulation includes colloid particles, you must specify in the input file
	(coll_x lines) the positions of the colloids at the last timestep of the previous
	simulation. These can be obtained from the final colloids file (columns 2-3 in 2D,
	2-4 in 3D).

4.- From the final micro file, create a file 'input_particles.in' with the following columns:

       --- 2D --

	   1.- x

	   2.- y

	   3.- Vel x

	   4.- Vel y

	   5.- mass

	   6.- type of particle

       --- 3D --

	   1.-  x

	   2.-  y

	   3.-  z	 

	   4.-  Vel x

	   5.-  Vel y

	   6.-  Vel z	

	   7.-  mass

	   8.-  type of particle

5.- Run the simulation

