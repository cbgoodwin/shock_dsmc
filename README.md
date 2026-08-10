# Shock DSMC

This repo implements the 1D Direct Simulation Monte Carlo for hypersonic rarefied gas flows in a tube.

## Getting Started

### Dependencies

* Python with standard packages (numpy, scipy, etc.)

## Description

The script `shock_simulation.py` simulates hypersonic rarefied gas flows in a one dimensional tube using the Hard Spheres collision model and a moving piston downstream boundary condition. In short, particles are injected with flux Maxwell distributed velocities at the injection (left) boundary, the particles collide and move through the tube according the the HS model, and eventually the particles specularly reflect off of the piston (right) boundary with x-component velocity 2Uw-c_x, where Uw is the downstream Rankine-Hugoniot bulk velocity and c_x is the prior x-component of velocity for the particle. Particles are removed if they remain right of the right boundary after a given time step.

## Scripts

### shock_simulation.py

This script executes the simulation. Optionally, the simulation can start where another simulation ended, but in general `shock_simulation.py` is a standalone script that does not require help from other scripts or data files in the repo.

#### Command Line Arguments
* `-warmup=n_w` tells the script to run the simulation for `n_w` time steps before recording any data. This was added to allow a burn-in phase since shocks are not necessarily initialized at a shape or location that is close to their equilibrium state.
* `-period=p`, `-num_periods=n_p`, `-gap=n_g` tells the script to take `n_p` snapshots (default=1) where each snapshot is an average over `p` consecutive time steps (default=10), and to run `g` time steps (default=0) between snapshots.
  * Example: Suppose the following command is run: `python shock_simulation.py -warmup=200 -period=10 -num_periods=20 -gap=100`. The simulation will run for 200 steps before recording any data. It will then alternate the following process 20 times: run 10 time steps and record/write average measurements of density, temperature, and bulk velocity over those 10 time steps, run 100 time steps (only recording number of particles in the tube and leaving at the right boundary). Thus, the simulation will run a total of 200 + 20\*10 + 19\*100 time steps.

* `-n=n_0` sets the overall simulated particle density in the tube to `n_0` (default=500). Thus, at initialization there will be n time the volume of the tube simulated particles in the tube (this does not apply if using `-initialize`).
* `-piston_speed=s` modifies the downstream piston boundary condition to use a frame of reference velocity of `s` times Uw (default=1).
*`-smooth_center=k` uses the average density profile of the trailing `k` time steps (default=3) as input to the shock center computation.
* `-seed=x` sets the seed for the random number generator, enabling comparison of matching simulations.
* `-initialize` can be set to the name of a folder containing a state file to pick up where another simulation left off, or the string 'empty' to initialize an empty tube. Default is to initialize particles in the tube according to upstream R-H relations for the left two thirds of the tube and downstream for the right third of the tube.
  
#### Execution
The first time this script is executed, there are no prior simulation state files to load so the below command (or something similar) can be run:
`python shock_simulation.py -warmup=200 -period=10 -num_periods=20 -gap=100`

A folder containing simulation data will be generated with a name like `ouput/shock_output_081026_123059` where the date and time in the folder name matches the simulation start date and time. The folder contains the following files:
* `rho.dat`, `T.dat`, `u.dat`: These files contain average density, temperature, and bulk velocity readings, respectively, along the length of the tube at each period. The values are an average over the length of the period for each period, so the above command will produce 20 sets of measurements, each set being the average over 10 timesteps.
* `npt.dat`, `removed_right.dat`: The total number of particles in the tube and number of particles removed at the right boundary are recorded for each time step after the warmup, including gap steps.
* `shock_center.dat`: The shock center is recorded for each time step after the warmup phase, including gap steps.
* `state.dat`: The final state of the simulation is saved in this file. This can then be used by a subsequent simulation to pick up where the previous simulation left off.
* `run_info.dat`: Information about the simulation run, including the command used to run the script and terminal output.

Once there is a folder in this format, another simulation can be run starting where the previous one left off:
`python shock_simulation.py -period=10 -num_periods=20 -gap=500 -initialize=output/shock_output_081026_123059`

A new folder containing simulation data will be generated, again with a name like `output/shock_output_081026_124110`.

### visualize_shock.py

This script generates visualizations for a single run of the shock simulation script. 

#### Command Line Arguments
fill in later

#### Execution
Run `python visualize_shock.py all` to generate plots for the most recent run. In particular, the data in the folder with name `shock_output_MMDDYY_hhmmss` and most recent date and time will be plotted. In order to generate plots for data in another folder, explicitly use the folder name, like this: `python visualize_shock all shock_output_081026_125423`.

### compare_center.py

This script overlays two plots of shock center over time for two separate runs. Both plots share the same x-axis, which is time. Each plot has a separate y-axis, although the width of each y-axis is equal so that the variances of the shock center plots are comparable.

#### Execution

The script needs to read two files in the format of the shock_center.dat output from shock_simulation.py. It will then display the plot and write the plot file to the same folder in which the two shock_center.dat files live.

* Add files `shock_center_1.dat` and `shock_center_2.dat` to the folder `shock_dsmc/compare_example`
* In directory `shock_dsmc`, run `python compare_center compare_example`
* The plot `compare_center.pdf` will be written to the folder `shock_dsmc/compare_example`

## Acknowledgments

The original Python code used for this project was a Claude translation of the Shock DSMC implemented in Fortran by someone else.
