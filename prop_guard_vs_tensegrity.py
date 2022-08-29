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

# Setup tensegrity
param = design_param()
tensegrity = tensegrity_design(param)
tensegrity.design_from_propeller()

nodeNum_t = tensegrity.nodeNum
dim_t = tensegrity.dim
joints_t = tensegrity.joints
rods_t = tensegrity.rods
numRod_t = tensegrity.numRod
strings_t = tensegrity.strings
numString_t = tensegrity.numString
massList_t = tensegrity.massList

P_t = np.zeros(nodeNum_t*dim_t*2) # Setup simulated values
# Rotate the whole tensegrity 90 degrees
tensegrityRot = Rotation.from_euler_YPR([-np.pi/2,0,0])
rotatedPos = np.zeros_like(tensegrity.nodePos)
for i in range(nodeNum_t):
    rotatedPos[i] = (tensegrityRot*Vec3(tensegrity.nodePos[i])).to_array().T
P_t[:nodeNum_t*dim_t] = rotatedPos.reshape((nodeNum_t*dim_t,))

# Setup prop_guard
# Create the prop-guard design
param = design_param()
prop_guard = prop_guard_design(param)
prop_guard.get_pos_from_propeller()
prop_guard.design()
nodeNum_p = prop_guard.nodeNum
dim_p = prop_guard.dim
joints_p = prop_guard.joints
links_p = prop_guard.links
massList_p = prop_guard.massList

# Rotate the whole tensegrity
P_p = np.zeros(nodeNum_p*dim_p*2) # Setup simulated values
propRot = Rotation.from_euler_YPR([np.pi/4,0,0])
rotatedPos = np.zeros_like(prop_guard.nodePosList)
for i in range(nodeNum_p):
    rotatedPos[i] = (propRot*Vec3(prop_guard.nodePosList[i])).to_array().squeeze()
P_p[:nodeNum_p*dim_p] = rotatedPos.reshape((nodeNum_p*dim_p,))

sectionNum = 3  #50

theta0 = np.linspace(-np.pi/2, np.pi/2, sectionNum)
theta1 = np.linspace(-np.pi/2, np.pi/2, sectionNum)

X, Y = np.meshgrid(theta0, theta1)

sizeTheta0 = theta0.shape[0]
sizeTheta1 = theta1.shape[0]

propGuardMaxStress = np.zeros((sizeTheta0,sizeTheta1))
tensegrityMaxStress = np.zeros((sizeTheta0,sizeTheta1))
totalF = 100 #[N]

for angleIdx0 in range (sectionNum):
    for angleIdx1 in range (sectionNum):
        print((angleIdx0,angleIdx1))
        angle0 = theta0[angleIdx0]
        angle1 = theta1[angleIdx1]
        print(angle0)
        print(angle1)
        rot =Rotation.from_euler_YPR([angle0,-angle1,0])
        print(rot*Vec3(1,0,0))        
        prop_guard_ODE =prop_guard_ode(prop_guard)
        sol_p = solve_ivp(prop_guard_ODE.ode_ivp_force_body_frame, t_span, P_p, method='Radau',args=(totalF,rot))
        tHist_p = sol_p.t
        Ps_p= sol_p.y
        np.savetxt("prop_t"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", sol_p.t, delimiter=",")
        np.savetxt("prop_P"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", sol_p.y, delimiter=",")

        stepCount_p = tHist_p.shape[0]
        nodePosHist_p = np.zeros((stepCount_p,nodeNum_p,dim_p))
        nodeVelHist_p = np.zeros((stepCount_p,nodeNum_p,dim_p))
        linkForceHist_p = np.zeros((stepCount_p,len(links_p)))
        nodeMaxStressHist_p = np.zeros((stepCount_p,nodeNum_p))
        prop_guard_helper = prop_guard_analysis(prop_guard_ODE)

        for i in range(nodeNum_p):
            for j in range(stepCount_p):
                nodePosHist_p[j,i,:] = Ps_p[i*dim_p:(i+1)*dim_p,j]
                nodeVelHist_p[j,i,:] = Ps_p[nodeNum_p*dim_p+i*dim_p:nodeNum_p*dim_p+(i+1)*dim_p,j]
        
        
        for i in range(stepCount_p):
            jointStress_i = prop_guard_helper.compute_joint_angle_and_torque(nodePosHist_p[i],nodeVelHist_p[i])[:,4]
            linkStress_i= prop_guard_helper.compute_link_stress(nodePosHist_p[i])
            nodeMaxStressHist_p[i] = prop_guard_helper.compute_node_max_stress(linkStress_i, jointStress_i)
        propGuardMaxStress[angleIdx0,angleIdx1]=np.max(nodeMaxStressHist_p.flatten())

        # Find max stress in tensegrity

        tensegrity_ODE =tensegrity_ode(tensegrity)
        sol_t = solve_ivp(tensegrity_ODE.ode_ivp_force_body_frame, t_span, P_t, method='Radau',args=(totalF,rot))
        tHist_t = sol_t.t
        Ps_t= sol_t.y
        np.savetxt("ten_t"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", sol_t.t, delimiter=",")
        np.savetxt("ten_P"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", sol_t.y, delimiter=",")
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
        tensegrityMaxStress[angleIdx0,angleIdx1]=np.max(nodeMaxStressHist_t.flatten())

# Creating figure
fig = plt.figure()
ax = plt.axes(projection="3d")

surf = ax.plot_surface(X, Y, propGuardMaxStress, cmap='Oranges',
                       linewidth=0, antialiased=False)

surf = ax.plot_surface(X, Y, tensegrityMaxStress, cmap='Blues',
                       linewidth=0, antialiased=False)

np.savetxt("maxPropGuardStress20.csv", propGuardMaxStress, delimiter=",")
np.savetxt("maxTensegrityStress20.csv", tensegrityMaxStress, delimiter=",")
np.savetxt("X20.csv", X, delimiter=",")
np.savetxt("Y20.csv", Y, delimiter=",")

ax.grid(False)
ax.set_xlabel('Angle 1 (rad)',fontsize=12)
ax.set_ylabel('Angle 2 (rad)',fontsize=12)
ax.set_zlabel('Max Stress (Pa)',fontsize=12)
ax.tick_params(axis='both', labelsize=10)
plt.show()
print("done")