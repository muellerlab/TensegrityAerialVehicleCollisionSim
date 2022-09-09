import numpy as np
import scipy as sp
from py3dmath.py3dmath import Rotation, Vec3
import matplotlib.pyplot as plt
from problem_setup import design_param

from prop_guard.design_prop_guard import *
from prop_guard.animate_prop_guard import *
from prop_guard.prop_guard_analysis import *
from prop_guard.prop_guard_ode import prop_guard_ode

from tensegrity.design_tensegrity import tensegrity_design
from tensegrity.tensegrity_ode import *
from tensegrity.tensegrity_analysis import *
from tensegrity.animate_tensegrity import *

from scipy.integrate import odeint, solve_ivp

import seaborn as sns
sns.set_theme()
# Setup simulation experiment
t0 = 0 # [s]
tf = 0.02 # [s] Simulation time
tF = tf # [s] Duration of collision
t_span = (t0,tf)
speed = 5 # speed of collision

sectionNum = 4
theta = np.linspace(0, np.pi/2, sectionNum)
preTension = np.linspace(0,100,sectionNum)
X, Y = np.meshgrid(theta, preTension)

sizeTheta = theta.shape[0]
sizePretension = preTension.shape[0]
tensegrityMaxStress = np.zeros((sizeTheta,sizePretension))

# Setup wall 
nWall = Vec3(1,0,0)
pWall = Vec3(0,0,0) #Set the wall so that contact starts right at time 0
Ew = 14e9 #[Pa], Young's modulus #https://www.engineeringtoolbox.com/concrete-properties-d_1223.html 
Aw = 0.1*0.1 #[m^2], effective area of compression
Lw = 3 #[m] thickness of wall
kWall = Ew*Aw/Lw #[N/m] Stiffness of wall


for thetaIdx in range (sectionNum):
    for preStressIdx in range (sectionNum):
        print("thetaIdx, preStressIdx =",(thetaIdx,preStressIdx))
        angle = X[thetaIdx,preStressIdx]
        att =Rotation.from_euler_YPR([0,-angle,0])
        # Setup tensegrity
        param = design_param() # get default parameter
        param.sPreT = Y[thetaIdx,preStressIdx] # Overwrite the prestress in string
        tensegrity = tensegrity_design(param)
        tensegrity.design_from_propeller()
        print("angle, preTension =",(angle,param.sPreT))

        nodeNum_t = tensegrity.nodeNum
        dim_t = tensegrity.dim
        joints_t = tensegrity.joints
        rods_t = tensegrity.rods
        numRod_t = tensegrity.numRod
        strings_t = tensegrity.strings
        numString_t = tensegrity.numString
        massList_t = tensegrity.massList

        tensegrityRot = Rotation.from_euler_YPR([-np.pi/2,0,0])
        defaultPos = np.zeros_like(tensegrity.nodePos)
        for i in range(nodeNum_t):
            defaultPos[i] = (tensegrityRot*Vec3(tensegrity.nodePos[i])).to_array().squeeze()

        # Rotate the vehicle to desired attitude
        initPos = np.zeros_like(tensegrity.nodePos)
        for i in range(nodeNum_t):
            initPos[i] = (att*Vec3(defaultPos[i])).to_array().squeeze()
        # Offset the vehicle so it is touching wall at the beginning of the simulation
        offset = np.min(initPos[:,0]) - 1e-10 # Horizontally offset the vehicle so it just starts to contact the wall at the begining of simulation.
        initVel = np.zeros_like(tensegrity.nodePos)  
        for i in range(nodeNum_t):
            initPos[i] = initPos[i] - offset * np.array([1,0,0])
            initVel[i] = speed*Vec3(-1,0,0).to_array().squeeze()

        P_t = np.zeros(nodeNum_t*dim_t*2) # Setup simulated values
        P_t[:nodeNum_t*dim_t] = initPos.reshape((nodeNum_t*dim_t,))
        P_t[nodeNum_t*dim_t:] = initVel.reshape((nodeNum_t*dim_t,))

        tensegrity_ODE =tensegrity_ode(tensegrity)
        sol_t = solve_ivp(tensegrity_ODE.ode_ivp_wall, t_span, P_t, method='Radau',args=(nWall, kWall, pWall),events=tensegrity_ODE.wall_check_simple)
        tHist_t = sol_t.t
        Ps_t= sol_t.y
        stepCount_t = tHist_t.shape[0]
        nodePosHist_t = np.zeros((stepCount_t,nodeNum_t,dim_t))
        nodeVelHist_t = np.zeros((stepCount_t,nodeNum_t,dim_t))
        rodStressHist_t = np.zeros((stepCount_t,len(tensegrity.rods)))
        nodeMaxStressHist_t = np.zeros((stepCount_t,nodeNum_t))
        for i in range(nodeNum_t):
            for j in range(stepCount_t):
                nodePosHist_t[j,i,:] = Ps_t[i*dim_t:(i+1)*dim_t,j]
                nodeVelHist_t[j,i,:] = Ps_t[nodeNum_t*dim_t+i*dim_t:nodeNum_t*dim_t+(i+1)*dim_t,j]
        
        tensegrity_helper = tensegrity_analysis(tensegrity_ODE)
        for i in range(stepCount_t):
            jointStress_i = tensegrity_helper.compute_joint_angle_and_torque(nodePosHist_t[i],nodeVelHist_t[i])[:,4]
            rodStressHist_t[i] = tensegrity_helper.compute_rod_stress(nodePosHist_t[i])
            nodeMaxStressHist_t[i] = tensegrity_helper.compute_node_max_stress(rodStressHist_t[i], jointStress_i)
        tensegrityMaxStress[thetaIdx,preStressIdx]=np.max(nodeMaxStressHist_t.flatten())
        print("maxStress=",tensegrityMaxStress[thetaIdx,preStressIdx])
        np.savetxt("preTS/t"+str(thetaIdx)+"_"+str(preStressIdx)+".csv", sol_t.t, delimiter=",")
        np.savetxt("preTS/p"+str(thetaIdx)+"_"+str(preStressIdx)+".csv", sol_t.y, delimiter=",")
        # Clean up objects
        tensegrity = None
        tensegrity_ODE = None
        tensegrity_helper = None

n = 20 # level of contours
vmin = np.min(tensegrityMaxStress)
vmax = np.max(tensegrityMaxStress)
levels = np.linspace(vmin, vmax, n+1)

fig1, ax = plt.subplots(constrained_layout=True)
cs = ax.contourf(theta, preTension, tensegrityMaxStress, cmap = 'GnBu',levels = levels)
ax.set_xlabel('pitch angle (rad)', fontsize=18)
ax.set_ylabel('string pre-tension (N)', fontsize=18)
ax.tick_params(labelsize=18)
ax.set_title('Max Stress: Tensegrity', fontsize=18)
fig1.colorbar(cs, orientation='horizontal')
plt.show()