import math
from platform import node
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

"""
Simulate the dynamics of a propeller guard vehicle under wall collision. 
Plot the information in the system and animate the process
"""
# Setup running functions
drawDebugPlots = True
createAnimation = True
slowMotionRate = 128 # Animate at 1/slowMotionRate of real speed 


# Create the prop-guard design
param = design_param()
prop_guard = prop_guard_design(param)
prop_guard.get_pos_from_propeller()
prop_guard.design()
nodeNum = prop_guard.nodeNum
dim = prop_guard.dim
joints = prop_guard.joints
rods = prop_guard.rods
crossJoints = prop_guard.crossJoints
massList = prop_guard.massList

# Setup simulation experiment
speed = 5 # speed of collision
t0 = 0 # [s]
tf = 0.02 # [s] Simulation time
t_span = (t0,tf)
t_eval = np.linspace(t0, tf, num = 300)
dt=t_eval[1]-t_eval[0] # Calculate step time. This is related to frame rate of animation.  


# Rotate the vehicle to default orientation: two propeller guards 
propRot = Rotation.from_euler_YPR([np.pi/4,0,0])
defaultPos = np.zeros_like(prop_guard.nodePosList)
for i in range(nodeNum):
    defaultPos[i] = (propRot*Vec3(prop_guard.nodePosList[i])).to_array().squeeze()

# Rotate the vehicle to desired attitude
att = Rotation.from_euler_YPR([0,-np.pi/4,0])
initPos = np.zeros_like(prop_guard.nodePosList)
for i in range(nodeNum):
    initPos[i] = (att*Vec3(defaultPos[i])).to_array().squeeze()

# Offset the vehicle so it is touching wall at the beginning of the simulation
offset = np.min(initPos[:,0]) - 1e-6 # Horizontally offset the vehicle so it just starts to contact the wall at the begining of simulation.
initVel = np.zeros_like(prop_guard.nodePosList)  
for i in range(nodeNum):
    initPos[i] = initPos[i] - offset * np.array([1,0,0])
    initVel[i] = speed*Vec3(-1,0,0).to_array().squeeze()

P0 = np.zeros(nodeNum*dim*2) # Setup simulated values
P0[:nodeNum*dim] = initPos.reshape((nodeNum*dim,))
P0[nodeNum*dim:] = initVel.reshape((nodeNum*dim,))

# Setup wall 
Ew = 14e9 #[Pa], Young's modulus #https://www.engineeringtoolbox.com/concrete-properties-d_1223.html 
Aw = 0.1*0.1 #[m^2], effective area of compression
Lw = 3 #[m] thickness of wall
nWall = Vec3(1,0,0)
pWall = Vec3(0,0,0) #Set the wall so that contact starts right at time 0
kWall = Ew*Aw/Lw #[N/m] Stiffness of wall

print("Simulation type: wall collision")
prop_guard_ODE =prop_guard_ode(prop_guard)
sol = solve_ivp(prop_guard_ODE.ode_ivp_wall, t_span, P0, method='Radau',args=(nWall, kWall, pWall), t_eval=t_eval)
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

crossJointAngleHist = np.zeros((stepCount,len(crossJoints)))
crossJointAngularRateHist = np.zeros((stepCount,len(crossJoints)))
crossJointSpringMomentHist = np.zeros((stepCount,len(crossJoints)))
crossJointDampingMomentHist = np.zeros((stepCount,len(crossJoints)))

rodStressHist = np.zeros((stepCount,len(rods)))
jointStressHist = np.zeros((stepCount,len(joints)))
crossJointStressHist = np.zeros((stepCount,len(crossJoints)))
nodeMaxStressHist = np.zeros((stepCount,nodeNum))
extForceHist = np.zeros((stepCount,nodeNum)) # external force for node collision

for i in range(nodeNum):
    for j in range(stepCount):
        nodePosHist[j,i,:] = Ps[i*dim:(i+1)*dim,j]
        nodeVelHist[j,i,:] = Ps[nodeNum*dim+i*dim:nodeNum*dim+(i+1)*dim,j]
for i in range(stepCount):
    jointInfo_i = prop_guard_helper.compute_joint_angle_and_torque(nodePosHist[i],nodeVelHist[i])
    crossJointInfo_i = prop_guard_helper.compute_cross_joint_angle_and_torque(nodePosHist[i],nodeVelHist[i])
    jointAngleHist[i,:] = jointInfo_i[:,0]
    jointAngularRateHist[i,:] = jointInfo_i[:,1]
    jointSpringMomentHist[i,:] = jointInfo_i[:,2]
    jointDampingMomentHist[i,:] = jointInfo_i[:,3]
    jointStressHist[i,:] = jointInfo_i[:,4]

    crossJointAngleHist[i,:] = crossJointInfo_i[:,0]
    crossJointAngularRateHist[i,:] = crossJointInfo_i[:,1]
    crossJointSpringMomentHist[i,:] = crossJointInfo_i[:,2]
    crossJointDampingMomentHist[i,:] = crossJointInfo_i[:,3]
    crossJointStressHist[i,:] = crossJointInfo_i[:,4]
    
    rodStressHist[i,:] = prop_guard_helper.compute_rod_stress(nodePosHist[i])
    nodeMaxStressHist[i,:] = prop_guard_helper.compute_node_max_stress(rodStressHist[i], jointStressHist[i,:],crossJointStressHist[i,:])
    extForceHist[i,:] = prop_guard_helper.compute_wall_collision_force(nodePosHist[i],nWall, kWall, pWall)

if drawDebugPlots:
    # Debug plots 
    fig = plt.figure()
    n = 1 # num sub-plots
    fig.add_subplot(n, 1, 1)

    for i in range(2, n + 1):
        fig.add_subplot(n, 1, i, sharex=fig.axes[0])

    for j in range(len(rods)):
        fig.axes[0].plot(tHist, rodStressHist[:,j], '-', label='Rod_'+str(j))
    fig.axes[0].set_ylabel('Stress [Pa]')
    fig.axes[0].set_xlabel('Time [s]')
    fig.axes[0].set_title('Stress in rods')
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
    fig2.axes[0].set_title('Joint spring moment')
    fig2.axes[1].set_title('Joint damping moment')

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
    fig3.axes[0].set_title('Node position')

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
    fig4.axes[0].set_title('Node velocity')


    fig5 = plt.figure()
    n = 2 # num sub-plots
    fig5.add_subplot(n, 1, 1)
    for i in range(2, n + 1):
        fig5.add_subplot(n, 1, i, sharex=fig5.axes[0])
    for j in range(len(joints)):
        fig5.axes[0].plot(tHist, jointAngleHist[:,j], '-', label='angleJoint'+str(j))
        fig5.axes[1].plot(tHist, jointAngularRateHist[:,j], '-', label='omegaJoint'+str(j))
    for j in range(len(crossJoints)):
        fig5.axes[0].plot(tHist, crossJointAngleHist[:,j], '-', label='angleCrossJoint'+str(j))
        fig5.axes[1].plot(tHist, crossJointAngularRateHist[:,j], '-', label='omegaCrossJoint'+str(j))

    fig5.axes[0].set_ylabel('Angle [Rad]')
    fig5.axes[1].set_ylabel('Angular Rate [Rad/s]')
    fig5.axes[0].set_xlabel('Time [s]')
    fig5.axes[0].legend()
    fig5.axes[1].legend()
    fig5.axes[0].set_title('Joint angle')
    fig5.axes[1].set_title('Joint angular velocity')

    fig6 = plt.figure()
    n = 2 # num sub-plots
    fig6.add_subplot(n, 1, 1)
    for i in range(2, n + 1):
        fig6.add_subplot(n, 1, i, sharex=fig6.axes[0])
    for j in range(len(joints)):
        fig6.axes[0].plot(tHist, jointStressHist[:,j], '-', label='stressJoint'+str(j))
    for j in range(len(crossJoints)):
        fig6.axes[0].plot(tHist, crossJointStressHist[:,j], '-', label='stressCrossJoint'+str(j))
    for j in range(nodeNum):
        fig6.axes[1].plot(tHist, nodeMaxStressHist[:,j], '-', label='maxStressNode'+str(j))
    fig6.axes[0].set_ylabel('Stress [Pa]')
    fig6.axes[1].set_ylabel('Stress [Pa]')
    fig6.axes[0].set_xlabel('Time [s]')
    fig6.axes[0].legend()
    fig6.axes[1].legend()
    fig6.axes[0].set_title('Stress in joint')
    fig6.axes[1].set_title('Max stress in node')

    # Fig7, external collison force from wall
    fig7 = plt.figure()
    n = 1 # num sub-plots
    fig7.add_subplot(n, 1, 1)
    for i in range(2, n + 1):
        fig7.add_subplot(n, 1, i, sharex=fig7.axes[0])
    for j in range(nodeNum):
        fig7.axes[0].plot(tHist, extForceHist[:,j], '-', label='extF'+str(j))
    fig7.axes[0].set_ylabel('Force [N]')
    fig7.axes[0].set_xlabel('Time [s]')
    fig7.axes[0].legend()
    fig7.axes[0].set_title('Wall collision force')

    if not(createAnimation):
        plt.show()

if createAnimation:
    fps = math.floor((1/dt)/slowMotionRate)
    print("Creating Animation")
    animator = prop_guard_animator(prop_guard)
    animator.plot_wall(pWall,'x')
    animator.animate_prop_guard(nodePosHist,True,True,fps)