import pickle
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
from scipy.integrate import solve_ivp
import seaborn as sns

from orientation_dist.uniform_orientation_dist import *

sns.set_theme()
# Folder to store results
folderName = "MonteCarloResult/"

# Set up the batch index and sample size for the Monte Carlo test.
# By default we generate 2000 samples for a single batch. 
batchIdx = 0
sampleSize = 10
scale = 1

# Setup wall 
nWall = Vec3(1,0,0)
Ew = 14e9 #[Pa], Young's modulus for concrete #https://www.engineeringtoolbox.com/concrete-properties-d_1223.html 
Aw = 0.1*0.1 #[m^2], effective area of compression
Lw = 3 #[m] thickness of wall
kWall = Ew*Aw/Lw #[N/m] Stiffness of wall
pWall = Vec3(0,0,0)

# Setup simulation experiment
t0 = 0 # [s]
tf = 0.05 # [s] Max simulation time
tF = tf # [s] Duration of collision
t_span = (t0,tf)
speed = 5 #[m/s] Initial speed 

# Setup tensegrity
param = design_param(scale)
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

# Rotate the whole tensegrity 90 degrees to get to the default collision orientation: two long rods hosting the drone is perpendicular ot the wall
tensegrityRot = Rotation.from_euler_YPR([-np.pi/2,0,0])
defaultPos_t = np.zeros_like(tensegrity.nodePos)
for i in range(nodeNum_t):
    defaultPos_t[i] = (tensegrityRot*Vec3(tensegrity.nodePos[i])).to_array().squeeze()

# Setup prop_guard
param = design_param()
prop_guard = prop_guard_design(param)
prop_guard.get_pos_from_propeller()
prop_guard.design()
nodeNum_p = prop_guard.nodeNum
dim_p = prop_guard.dim
joints_p = prop_guard.joints
rods_p = prop_guard.rods
massList_p = prop_guard.massList

# Rotate the whole prop-guard to default orientation: two neighboring propellers are forming a line parallel to the wall.
P_p = np.zeros(nodeNum_p*dim_p*2) # Setup simulated values
propRot = Rotation.from_euler_YPR([np.pi/4,0,0])
defaultPos_p = np.zeros_like(prop_guard.nodePosList)
for i in range(nodeNum_p):
    defaultPos_p[i] = (propRot*Vec3(prop_guard.nodePosList[i])).to_array().squeeze()

thetaRange = [np.pi/2,np.pi]
phiRange = [0,np.pi/2]
vCollision = Vec3([-1,0,0])
orientDist = OrientDist(thetaRange,phiRange,vCollision,batchIdx) #Use the batch index as seed
sampleOrientation = orientDist.uniform_sample(sampleSize)

sampledOrientation=dict({"orientation":sampleOrientation})
file = open(folderName+"sampledOrientations"+"_"+str(batchIdx)+".pickle", 'wb')
pickle.dump(sampledOrientation, file)
file.close()

for idx in range (sampleSize):
    y = sampleOrientation[idx,3]
    p = sampleOrientation[idx,4]
    r = sampleOrientation[idx,5]
    print("id="+str(idx)+":["+str(y)+","+str(p)+","+str(r)+"]") 
    att = Rotation.from_euler_YPR([y,p,r])
    
    P_p = np.zeros(nodeNum_p*dim_p*2) # Setup simulated values
    initPos_p = np.zeros_like(defaultPos_p)
    initVel_p = np.zeros_like(initPos_p)
    for i in range(nodeNum_p):
        initPos_p[i] = (att*Vec3(defaultPos_p[i])).to_array().squeeze()
    
    # Offset the vehicle so it is touching wall at the beginning of the simulation
    offset_p = np.min(initPos_p[:,0]) - 1e-6 
    for i in range(nodeNum_p):
        initPos_p[i] = initPos_p[i]- offset_p * np.array([1,0,0])
        initVel_p[i] = speed*Vec3(-1,0,0).to_array().squeeze()
    P_p[:nodeNum_p*dim_p] = initPos_p.reshape((nodeNum_p*dim_p,))
    P_p[nodeNum_p*dim_p:] = initVel_p.reshape((nodeNum_p*dim_p,))        
    
    prop_guard_ODE =prop_guard_ode(prop_guard)
    sol_p = solve_ivp(prop_guard_ODE.ode_ivp_wall, t_span, P_p, method='Radau',args=(nWall, kWall, pWall),events=prop_guard_ODE.vel_check_simple)
    tHist_p = sol_p.t
    Ps_p= sol_p.y

    simResult_p=dict({"P":Ps_p, "t":tHist_p, "attYPR":np.array([y,p,r])}) 
    file = open(folderName+"prop"+str(batchIdx)+"_"+str(idx)+".pickle", 'wb')
    pickle.dump(simResult_p, file)
    file.close()

    # Find max stress in tensegrity
    P_t = np.zeros(nodeNum_t*dim_t*2) # Setup simulated values
    initPos_t = np.zeros_like(defaultPos_t)
    initVel_t = np.zeros_like(initPos_t)
    for i in range(nodeNum_t):
        initPos_t[i] = (att*Vec3(defaultPos_t[i])).to_array().squeeze()
    offset_t = np.min(initPos_t[:,0]) - 1e-6 # Horizontally offset the vehicle so it just starts to contact the wall at the begining of simulation.
    for i in range(nodeNum_t):
        initPos_t[i] = initPos_t[i] - offset_t * np.array([1,0,0])
        initVel_t[i] = speed*Vec3(-1,0,0).to_array().squeeze()
    P_t[:nodeNum_t*dim_t] = initPos_t.reshape((nodeNum_t*dim_t,))
    P_t[nodeNum_t*dim_t:] = initVel_t.reshape((nodeNum_t*dim_t,))        

    tensegrity_ODE =tensegrity_ode(tensegrity)
    sol_t = solve_ivp(tensegrity_ODE.ode_ivp_wall, t_span, P_t, method='Radau',args=(nWall, kWall, pWall),events=tensegrity_ODE.vel_check_simple)
    tHist_t = sol_t.t
    Ps_t= sol_t.y

    simResult=dict({"P":Ps_t, "t":tHist_t,"attYPR":np.array([y,p,r])}) 
    file = open(folderName+"ten"+str(batchIdx)+"_"+str(idx)+".pickle", 'wb')
    pickle.dump(simResult, file)
    file.close()