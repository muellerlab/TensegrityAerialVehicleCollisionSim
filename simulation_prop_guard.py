import numpy as np
import scipy as sp
from py3dmath.py3dmath import Rotation, Vec3
import matplotlib.pyplot as plt
from problem_setup import design_param
from prop_guard.design_prop_guard import *
from prop_guard.animate_prop_guard import *
from prop_guard.prop_guard_analysis import *
from prop_guard.prop_guard_ode import prop_guard_ode
from scipy.integrate import odeint, solve_ivp

import seaborn as sns
sns.set_theme()

# Create the prop-guard design
param = design_param()
prop_guard = prop_guard_design(param)
prop_guard.get_pos_from_propeller()
prop_guard.design()
nodeNum = prop_guard.nodeNum
dim = prop_guard.dim
joints = prop_guard.joints
links = prop_guard.links

# Setup simulation experiment
t0 = 0 # [s]
tf = 0.05 # [s] Simulation time
tF = tf # [s] Duration of collision
t_span = (t0,tf)
theta = np.pi/4 #[rad] angle of collision
hitForce = 20 #[N]

# Rotate the whole tensegrity
P0 = np.zeros(nodeNum*dim*2) # Setup simulated values
propRot = Rotation.from_euler_YPR([np.pi/4,0,0])
rotatedPos = np.zeros_like(prop_guard.nodePosList)
for i in range(nodeNum):
    rotatedPos[i] = (propRot*Vec3(prop_guard.nodePosList[i])).to_array().squeeze()
P0[:nodeNum*dim] = rotatedPos.reshape((nodeNum*dim,))

# Define external collision force
extF = np.zeros((nodeNum,dim))
F = hitForce/2 * np.array([np.cos(theta), 0, np.sin(theta)]) 
extF[0] += F
extF[8] += F

# Define and solve ODE
print("Start Simulation")
prop_guard_ODE =prop_guard_ode(prop_guard)
sol = solve_ivp(prop_guard_ODE.ode_ivp_hit_force, t_span, P0, method='Radau', args = (extF, tF))
print("Finish Simulation")


# Analyze the ODE result
print("Recording Results")
prop_guard_helper = prop_guard_analysis(prop_guard_ODE)
tHist = sol.t
Ps = sol.y
stepCount = tHist.shape[0]
nodePosHist = np.zeros((stepCount,nodeNum,dim))
nodeVelHist = np.zeros((stepCount,nodeNum,dim))
jointAngleHist = np.zeros((stepCount,len(joints)))
jointAngularRateHist = np.zeros((stepCount,len(joints)))
jointSpringMomentHist = np.zeros((stepCount,len(joints)))
jointDampingMomentHist = np.zeros((stepCount,len(joints)))
jointStressHist = np.zeros((stepCount,len(joints)))
linkStressHist = np.zeros((stepCount,len(links)))
nodeMaxStressHist = np.zeros((stepCount,nodeNum))

for i in range(nodeNum):
    for j in range(stepCount):
        nodePosHist[j,i,:] = Ps[i*dim:(i+1)*dim,j]
        nodeVelHist[j,i,:] = Ps[nodeNum*dim+i*dim:nodeNum*dim+(i+1)*dim,j]
for i in range(stepCount):
    jointInfo_i = prop_guard_helper.compute_joint_angle_and_torque(nodePosHist[i],nodeVelHist[i])
    jointAngleHist[i,:] = jointInfo_i[:,0]
    jointAngularRateHist[i,:] = jointInfo_i[:,1]
    jointSpringMomentHist[i,:] = jointInfo_i[:,2]
    jointDampingMomentHist[i,:] = jointInfo_i[:,3]
    jointStressHist[i,:] = jointInfo_i[:,4]
    linkStressHist[i,:] = prop_guard_helper.compute_link_stress(nodePosHist[i])
    nodeMaxStressHist[i,:] = prop_guard_helper.compute_node_max_stress(linkStressHist[i], jointStressHist[i,:])

# Plot used in paper 
figp = plt.figure(figsize=(36,7))
n = 1 # num sub-plots
figp.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    figp.add_subplot(n, 1, i, sharex=figp.axes[0])

figp.axes[0].plot(tHist, nodeMaxStressHist[:,joints[0][1]], 'r-', label='Joint0&3', linewidth=6)
figp.axes[0].plot(tHist, nodeMaxStressHist[:,joints[1][1]], 'b-', label='Joint1&4', linewidth=6)
figp.axes[0].plot(tHist, nodeMaxStressHist[:,joints[2][1]], 'y-', label='Joint2', linewidth=6)
figp.axes[0].set_ylabel('Stress in the propeller guard frame [Pa]', fontsize=25)
figp.axes[0].set_xlabel('Time [s]', fontsize=25)
figp.axes[0].legend(fontsize=20)
figp.axes[0].tick_params(axis='x', labelsize=20)
figp.axes[0].tick_params(axis='y', labelsize=20)
figp.axes[0].yaxis.get_offset_text().set_fontsize(20)

# Debug plots 
fig = plt.figure()
n = 1 # num sub-plots
fig.add_subplot(n, 1, 1)

for i in range(2, n + 1):
    fig.add_subplot(n, 1, i, sharex=fig.axes[0])

for j in range(len(links)):
    fig.axes[0].plot(tHist, linkStressHist[:,j], '-', label='Link_'+str(j))
fig.axes[0].set_ylabel('Stress [Pa]')
fig.axes[0].set_xlabel('Time [s]')
fig.axes[0].legend()

fig2 = plt.figure()
n = 2 # num sub-plots
fig2.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    fig2.add_subplot(n, 1, i, sharex=fig2.axes[0])
for j in range(len(joints)):
    fig2.axes[0].plot(tHist, jointSpringMomentHist[:,j], '-', label='bendingMomentJoint_'+str(j))
    fig2.axes[1].plot(tHist, jointDampingMomentHist[:,j], '-', label='dampingMomentJoint_'+str(j))
fig2.axes[0].set_ylabel('Moment [Nm]')
fig2.axes[1].set_ylabel('Moment [Nm]')
fig2.axes[0].set_xlabel('Time [s]')
fig2.axes[0].legend()
fig2.axes[1].legend()

fig3 = plt.figure()
n = nodeNum # num sub-plots
fig3.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    fig3.add_subplot(n, 1, i, sharex=fig3.axes[0])

for j in range(nodeNum):
    fig3.axes[j].plot(tHist, nodePosHist[:,j,0], 'r-', label='pos'+str(j)+'x')
    fig3.axes[j].plot(tHist, nodePosHist[:,j,1], 'g-', label='pos'+str(j)+'y')
    fig3.axes[j].plot(tHist, nodePosHist[:,j,2], 'b-', label='pos'+str(j)+'z')
    fig3.axes[j].set_ylabel('Pos [m]')
    fig3.axes[j].set_xlabel('Time [s]')
    fig3.axes[j].legend()

fig4 = plt.figure()
n = nodeNum # num sub-plots
fig4.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    fig4.add_subplot(n, 1, i, sharex=fig4.axes[0])

for j in range(nodeNum):
    fig4.axes[j].plot(tHist, nodeVelHist[:,j,0], 'r-', label='vel'+str(j)+'x')
    fig4.axes[j].plot(tHist, nodeVelHist[:,j,1], 'g-', label='vel'+str(j)+'y')
    fig4.axes[j].plot(tHist, nodeVelHist[:,j,2], 'b-', label='vel'+str(j)+'z')
    fig4.axes[j].set_ylabel('Vel [m/s]')
    fig4.axes[j].set_xlabel('Time [s]')
    fig4.axes[j].legend()

fig5 = plt.figure()
n = 2 # num sub-plots
fig5.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    fig5.add_subplot(n, 1, i, sharex=fig5.axes[0])
for j in range(len(joints)):
    fig5.axes[0].plot(tHist, jointAngleHist[:,j], '-', label='angleJoint'+str(j))
    fig5.axes[1].plot(tHist, jointAngularRateHist[:,j], '-', label='omegaJoint'+str(j))
fig5.axes[0].set_ylabel('Angle [Rad]')
fig5.axes[1].set_ylabel('Angular Rate [Rad/s]')
fig5.axes[0].set_xlabel('Time [s]')
fig5.axes[0].legend()
fig5.axes[1].legend()

fig6 = plt.figure()
n = 2 # num sub-plots
fig6.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    fig6.add_subplot(n, 1, i, sharex=fig5.axes[0])
for j in range(len(joints)):
    fig6.axes[0].plot(tHist, jointStressHist[:,j], '-', label='stressJoint'+str(j))

for j in range(nodeNum):
    fig6.axes[1].plot(tHist, nodeMaxStressHist[:,j], '-', label='maxStressNode'+str(j))
fig6.axes[0].set_ylabel('Stress [Pa]')
fig6.axes[1].set_ylabel('Stress [Pa]')
fig6.axes[0].set_xlabel('Time [s]')
fig6.axes[0].legend()
fig6.axes[1].legend()


frameSampleRate = 100 # use 1 frame for each 100 samples 
animateIteration = stepCount//frameSampleRate
nodePosAnimateData = np.zeros((animateIteration,nodeNum,dim))
for i in range(animateIteration):
    nodePosAnimateData[i] = nodePosHist[i*frameSampleRate]

print("Creating Animation")
animator = prop_guard_animator(prop_guard)
animator.animate_prop_guard(nodePosHist,True,False)
