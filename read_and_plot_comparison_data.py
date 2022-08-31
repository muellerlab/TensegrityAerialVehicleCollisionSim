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

folderName = "compareStudyResult/" # Name of the folder storing the data

# Setup tensegrity info
param = design_param()
tensegrity = tensegrity_design(param)
tensegrity.design_from_propeller()

nodeNum_t = tensegrity.nodeNum
dim_t = tensegrity.dim

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

sectionNum = 30  #50
theta0 = np.linspace(0, np.pi/2, sectionNum)
theta1 = np.linspace(0, np.pi/2, sectionNum)
X = np.zeros((sectionNum,sectionNum))
Y = np.zeros((sectionNum,sectionNum))
sizeTheta0 = theta0.shape[0]
sizeTheta1 = theta1.shape[0]

propGuardMaxStress = np.zeros((sizeTheta0,sizeTheta1))
tensegrityMaxStress = np.zeros((sizeTheta0,sizeTheta1))

prop_guard_ODE =prop_guard_ode(prop_guard)
tensegrity_ODE =tensegrity_ode(tensegrity)

for angleIdx0 in range (sizeTheta0):
    for angleIdx1 in range (sizeTheta1):
        X[angleIdx0,angleIdx1] = theta0[angleIdx0]
        Y[angleIdx0,angleIdx1] = theta1[angleIdx1]

        print("TestID:",(angleIdx0,angleIdx1))
        Ps_p = np.genfromtxt(folderName+"prop_P"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", delimiter=',')
        tHist_p = np.genfromtxt(folderName+"prop_t"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", delimiter=',')
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
        Ps_t = np.genfromtxt(folderName+"ten_P"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", delimiter=',')
        tHist_t = np.genfromtxt(folderName+"ten_t"+str(angleIdx0)+"_"+str(angleIdx1)+".csv", delimiter=',')

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

print(np.mean(propGuardMaxStress))
print(np.mean(tensegrityMaxStress))

plotType = "heatmap"
if plotType == "heatmap":
    # Creating figure
    n = 600 # level of contours
    vmin = min([np.min(propGuardMaxStress),np.min(tensegrityMaxStress)])
    vmax = max([np.max(propGuardMaxStress),np.max(tensegrityMaxStress)])
    levels = np.linspace(vmin, vmax, n+1)


    fig1, axs = plt.subplots(nrows=1,ncols=2, figsize=(8,8),sharex='col', sharey='row')
    (ax1, ax2) = axs

    levels = np.linspace(vmin, vmax, n+1)

    cs1 = ax1.contourf(X, Y, propGuardMaxStress, cmap = 'plasma',levels = levels)
    ax1.set_ylabel('Initial Pitch (rad)', fontsize=18)
    ax1.set_xlabel('Initial Yaw (rad)', fontsize=18)
    ax1.tick_params(labelsize=18)
    ax1.set_title('Max Stress:Prop-guard', fontsize=18)
    ax1.set_aspect('equal')

    cs2 = ax2.contourf(X, Y, tensegrityMaxStress, cmap = 'plasma',levels = levels)
    ax2.set_xlabel('Initial Yaw (rad)', fontsize=18)
    ax2.tick_params(labelsize=18)
    ax2.set_title('Max Stress:Tensegrity', fontsize=18)
    ax2.set_aspect('equal')
    fig1.colorbar(cs2, ax=axs.ravel().tolist(),orientation='horizontal')
    plt.show()

elif plotType == "3d":
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    surf = ax.plot_surface(X, Y, propGuardMaxStress, cmap='Oranges',
                        linewidth=0, antialiased=False)

    surf = ax.plot_surface(X, Y, tensegrityMaxStress, cmap='Blues',
                        linewidth=0, antialiased=False)

    # np.savetxt("maxPropGuardStress20.csv", propGuardMaxStress, delimiter=",")
    # np.savetxt("maxTensegrityStress20.csv", tensegrityMaxStress, delimiter=",")
    # np.savetxt("X20.csv", X, delimiter=",")
    # np.savetxt("Y20.csv", Y, delimiter=",")

    ax.grid(False)
    ax.set_xlabel('Initial Yaw (rad)',fontsize=12)
    ax.set_ylabel('Initial Pitch (rad)',fontsize=12)
    ax.set_zlabel('Max Stress (Pa)',fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    plt.show()
    print("done")