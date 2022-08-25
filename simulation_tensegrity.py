import numpy as np
from py3dmath.py3dmath import *
import matplotlib.pyplot as plt
from problem_setup import design_param
from tensegrity.design_tensegrity import tensegrity_design
from tensegrity.tensegrity_ode import *
from tensegrity.tensegrity_analysis import *
from tensegrity.animate_tensegrity import *
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set_theme()

"""
Simulate the dynamics of a single tensegrity under a sudden external load.
Plot the forces in the system and animate the process
"""

# Setup tensegrity
param = design_param()
tensegrity = tensegrity_design(param)
tensegrity.design_from_propeller()
nodeNum = tensegrity.nodeNum

dim = tensegrity.dim
joints = tensegrity.joints
rods = tensegrity.rods
numRod = tensegrity.numRod
strings = tensegrity.strings
numString = tensegrity.numString

# Setup simulation experiment
t0 = 0 # [s]
tf = 0.001 # [s] Simulation time
tF = tf # [s] Duration of collision
t_span = (t0,tf)

P0 = np.zeros(nodeNum*dim*2) # Setup simulated values
# Rotate the whole tensegrity
tensegrityRot = Rotation.from_euler_YPR([-np.pi/2,0,0])
rotatedPos = np.zeros_like(tensegrity.nodePos)
for i in range(nodeNum):
    rotatedPos[i] = (tensegrityRot*Vec3(tensegrity.nodePos[i])).to_array().T
P0[:nodeNum*dim] = rotatedPos.reshape((nodeNum*dim,))

theta = np.pi/4 #[rad] angle of collision
hitForce = 10 #[N]

extF = np.zeros((nodeNum,dim))
F = hitForce/2*np.array([np.cos(theta),0,np.sin(theta)])
extF[4]+=F
extF[6]+=F

# Define and solve ODE
print("Start Simulation")
tensegrity_ODE =tensegrity_ode(tensegrity)
sol = solve_ivp(tensegrity_ODE.ode_ivp_hit_force, t_span, P0, method='Radau', args = (extF, tF))
print("Finish Simulation")

# Analyze the ODE result
print("Recording Results")
tensegrity_helper = tensegrity_analysis(tensegrity_ODE)

tHist = sol.t 
Ps = sol.y
stepCount = tHist.shape[0]
nodePosHist = np.zeros((stepCount,nodeNum,dim))
nodeVelHist = np.zeros((stepCount,nodeNum,dim))
stringForceHist = np.zeros((stepCount,numString))
rodForceHist =  np.zeros((stepCount,numRod))
rodStressHist = np.zeros((stepCount,numRod))
nodeElasticForceHist = np.zeros((stepCount,nodeNum,dim))
nodeDampingForceHist = np.zeros((stepCount,nodeNum,dim))
nodeRotationForceHist = np.zeros((stepCount,nodeNum,dim)) 

jointAngleHist = np.zeros((stepCount,len(joints)))
jointAngularRateHist = np.zeros((stepCount,len(joints)))
jointSpringMomentHist = np.zeros((stepCount,len(joints)))
jointDampingMomentHist = np.zeros((stepCount,len(joints)))
jointStressHist = np.zeros((stepCount,len(joints)))
nodeMaxStressHist = np.zeros((stepCount,nodeNum))

for i in range(nodeNum):
    for j in range(stepCount):
        nodePosHist[j,i,:] = Ps[i*dim:(i+1)*dim,j]
        nodeVelHist[j,i,:] = Ps[nodeNum*dim+i*dim:nodeNum*dim+(i+1)*dim,j]
for i in range(stepCount):
    stringForceHist[i] = tensegrity_helper.compute_string_force(nodePosHist[i])
    rodForceHist[i] = tensegrity_helper.compute_rod_force(nodePosHist[i])
    rodStressHist[i] = rodForceHist[i]/tensegrity.rA
    stepForce = tensegrity_ODE.compute_internal_forces(Ps[:,i], True)
    nodeElasticForceHist[i,:,:] = stepForce[0]
    nodeDampingForceHist[i,:,:] = stepForce[1]
    nodeRotationForceHist[i,:,:] = stepForce[2]

    jointInfo_i = tensegrity_helper.compute_joint_angle_and_torque(nodePosHist[i],nodeVelHist[i])
    jointAngleHist[i,:] = jointInfo_i[:,0]
    jointAngularRateHist[i,:] = jointInfo_i[:,1]
    jointSpringMomentHist[i,:] = jointInfo_i[:,2]
    jointDampingMomentHist[i,:] = jointInfo_i[:,3]
    jointStressHist[i,:] = jointInfo_i[:,4]
    nodeMaxStressHist[i,:] = tensegrity_helper.compute_node_max_stress(rodStressHist[i], jointStressHist[i,:])
    
# Fig used in paper 
figp = plt.figure(figsize=(36,7))
n = 1 # num sub-plots
figp.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    figp.add_subplot(n, 1, i, sharex=figp.axes[0])

figp.axes[0].plot(tHist, jointStressHist[:,0], 'r-', label='Joint0&2 ', linewidth=6)
figp.axes[0].plot(tHist, jointStressHist[:,1], 'b-', label='Joint1&3 ', linewidth=6)
figp.axes[0].set_ylabel('Tensegrity Bending Stress [Pa]', fontsize=25)
figp.axes[0].set_xlabel('Time [s]', fontsize=25)
figp.axes[0].legend(fontsize=20)
figp.axes[0].tick_params(axis='x', labelsize=20)
figp.axes[0].tick_params(axis='y', labelsize=20)
figp.axes[0].yaxis.get_offset_text().set_fontsize(20)

# Fig used for debug
fig = plt.figure()
n = 2 # num sub-plots
fig.add_subplot(n, 1, 1)

for i in range(2, n + 1):
    fig.add_subplot(n, 1, i, sharex=fig.axes[0])

for j in range(24):
    fig.axes[0].plot(tHist, stringForceHist[:,j], '-', label='fString_'+str(j))
    fig.axes[0].set_ylabel('Force [N]')
    fig.axes[0].set_xlabel('Time [s]')
    fig.axes[0].legend()

for j in range(6):
    fig.axes[1].plot(tHist, rodForceHist[:,j], '-', label='fRod_'+str(j))
    fig.axes[1].set_ylabel('Force [N]')
    fig.axes[1].set_xlabel('Time [s]')
    fig.axes[1].legend()

fig2 = plt.figure()
n = 2 # num sub-plots
fig2.add_subplot(n, 1, 1)
for i in range(2, n + 1):
    fig2.add_subplot(n, 1, i, sharex=fig2.axes[0])

for j in range(len(joints)):
    fig2.axes[0].plot(tHist, jointSpringMomentHist[:,j], '-', label='Joint '+str(j), linewidth=3)
    fig2.axes[1].plot(tHist, jointDampingMomentHist[:,j], '-', label='dampingMomentJoint_'+str(j))
fig2.axes[0].set_ylabel('Moment [Nm]')
fig2.axes[1].set_ylabel('Moment [Nm]')
fig2.axes[1].set_xlabel('Time [s]')
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

# Animation
frameSampleRate = 10 #use 1 frame for each 1 sample 
animateIteration = stepCount//frameSampleRate
nodePosAnimateData = np.zeros((animateIteration,nodeNum,dim))
for i in range(animateIteration):
    nodePosAnimateData[i] = nodePosHist[i*frameSampleRate]

print("Creating Animation")
animator = tensegrity_animator(tensegrity)
animator.animate_tensegrity(nodePosHist,True,False)