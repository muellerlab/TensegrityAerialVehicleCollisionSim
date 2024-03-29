# Tensegrity Aerial Vehicle Collision Simulator

This repository contains source code implementing dynamics simulation of a tensegrity aerial vehicle colliding with a flat obstacle, such as a concrete wall. The code can be used to verify the design of icosahedron protection shells protecting aerial vehicles in the form of quadcopters. 

The detail of the simulation is described in the paper "Design and control of a collision-resilient aerial vehicle with an icosahedron tensegrity structure" submitted to  IEEE/ASME Transactions on Mechatronics (TMECH). A manuscript draft can be accessed [here](https://hiperlab.berkeley.edu/wp-content/uploads/2023/06/TensegrityAerialVehicle.pdf)). 



This work is evolved from our previous [IROS 2020 paper](https://ieeexplore.ieee.org/document/9341236).

Contact: Clark Zha (clark.zha@berkeley.edu)
High Performance Robotics Lab, Dept. of Mechanical Engineering, UC Berkeley


## Dependencies
The code uses following common python packages:
```
numpy, scipy, matplotlib, seaborn
```
If you want to generate videos for your simulation result, you additionally need:
```
ffmpeg
```
In addition, the code uses [py3dmath](https://github.com/muellerlab/TensegrityAerialVehicleCollisionSim) for 3D vector computation. For the ease of usage, we include a copy of the package in this repository so no additional installation is required.  

We also provide a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment file to help with the environment setup process. Simply run
```
conda env create -f environment.yml
```
with the provided yml file in your terminal to setup a proper python environment to run the code. 

## Quick Start Guide

To visualize a tensegrity vehicle structure:
```
python plot_vehicles.py
```

To visualize a tensegrity vehicle side by side with a propeller-guard vehicle for comparison:
```
python plot_tensegrity_vehicle.py
```

To run an example simulation of a tensegrity vehicle colliding into a wall: 
```
python sim_tensegrity_wall_collision.py
```

In addition, the "Tensegrity VS Propeller-guard Monte Carlo Study" in the paper can be conducted by running:
```
mkdir MonteCarloResult
mkdir AnalysisResultPaper
python monte_carlo_experiment.py
python monte_carlo_prepare_data.py
```
Notice that you need write permission to the the generated folders to store pickle files generated by the simulation code. In addition, this will take a relatively long computation time as more than 2000 (possibly stiff) ODEs with non-trivial dimensions are solved. We recommend you to start with a smaller scale experiment by changing the parameter "sampleSize" in both python scripts. 

## Visualization of Monte Carlo Analysis Result

We have provided the data for analysis used in paper in the ```/AnalysisResultPaper``` folder. 
To plot the data, run 
```
python monte_carlo_analysis.py
python monte_carlo_scale_analysis.py
```
to generate the results of maximum stress analysis and scale analysis. 

Notice that by setting ```usePreparedData = False``` in ```python monte_carlo_analysis.py```, you can use the newly created data generated in the previous step, instead of the provided data.

## Using this software as tensegrity design tool 
To create and test your own tensegrity vehicle, you can change parameters like size, weight, material, etc. in 
```
problem_setup.py
```
You can also adjust initial orientation and speed of the tesegrity in the wall collision experiment by modifying the setup section in
```
simulation_tensegrity_wall_collision.py
```


## Accompanying video
[![Watch the video](https://user-images.githubusercontent.com/39609430/204390127-ffbfe43a-ce01-471d-992a-9bfe98cb9b22.png)](https://youtu.be/XsLVRd2nMd0)


## Acknowledgement
Co-authors of the paper: Xiangyu Wu, Ryan Dimick, Mark. W. Mueller

Collaborators who have contributted to the tensegrity aerial vehicle developement: Joey Kroeger, Natalia Perez 

Scholars who have provided their insights on the tensegrity aerial vehicle: Alice Agogino, Alan Zhang, Douglas Hutchings, Kévin Garanger
