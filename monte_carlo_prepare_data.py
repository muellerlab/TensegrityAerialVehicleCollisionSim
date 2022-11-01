import numpy as np
import scipy as sp
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
import pickle
import matplotlib.colors as colors
import seaborn as sns
sns.set_theme()
folderName = "MonteCarloResult/" # Name of the folder read the simulation data from
saveFolderName = "AnalysisResult/" # Name of the folder store the simulation data to. 
plotData = False

# Setup the Monte Carlo Test Information 
# Default test is 1 batch of 2000 samples. 
scale = 1 # scale factor for the Monte Carlo test.
scaleText = "1"
batchNum = 1 
sampleNum = 10
totalNum = batchNum*sampleNum
param = design_param(scale)

# Setup tensegrity info
tensegrity = tensegrity_design(param)
tensegrity.design_from_propeller()
nodeNum_t = tensegrity.nodeNum
dim_t = tensegrity.dim

# Setup prop guard info
prop_guard = prop_guard_design(param)
prop_guard.get_pos_from_propeller()
prop_guard.design()
nodeNum_p = prop_guard.nodeNum
dim_p = prop_guard.dim
joints_p = prop_guard.joints
rods_p = prop_guard.rods

prop_guard_ODE =prop_guard_ode(prop_guard)
tensegrity_ODE =tensegrity_ode(tensegrity)

propGuardData = np.zeros((totalNum,7)) #track 7 variables [x,y,z,y,p,r,maxStress]
tensegrityData = np.zeros((totalNum,7)) #track 7 variables [x,y,z,y,p,r,maxStress]

for batchIdx in range(batchNum):
    file_O = open(folderName+"sampledOrientations"+"_"+str(batchIdx)+".pickle", 'rb')
    data_O = pickle.load(file_O)
    file_O.close()
    orientationSample = data_O["orientation"]
    
    for sampleIdx in range(sampleNum):
        print("TestID:",[batchIdx,sampleIdx])
        file_p = open(folderName+"prop"+str(batchIdx)+"_"+str(sampleIdx)+".pickle", 'rb')
        data_p = pickle.load(file_p)
        file_p.close()
        Ps_p = data_p["P"]
        tHist_p = data_p["t"]
        YPR_p = data_p["attYPR"]

        stepCount_p = tHist_p.shape[0]
        nodePosHist_p = np.zeros((stepCount_p,nodeNum_p,dim_p))
        nodeVelHist_p = np.zeros((stepCount_p,nodeNum_p,dim_p))
        rodForceHist_p = np.zeros((stepCount_p,len(rods_p)))
        nodeMaxStressHist_p = np.zeros((stepCount_p,nodeNum_p))
        prop_guard_helper = prop_guard_analysis(prop_guard_ODE)

        for i in range(nodeNum_p):
            for j in range(stepCount_p):
                nodePosHist_p[j,i,:] = Ps_p[i*dim_p:(i+1)*dim_p,j]
                nodeVelHist_p[j,i,:] = Ps_p[nodeNum_p*dim_p+i*dim_p:nodeNum_p*dim_p+(i+1)*dim_p,j]
        
        for i in range(stepCount_p):
            jointStress_i = prop_guard_helper.compute_joint_angle_and_torque(nodePosHist_p[i],nodeVelHist_p[i])[:,4]
            crossJointStress_i = prop_guard_helper.compute_cross_joint_angle_and_torque(nodePosHist_p[i],nodeVelHist_p[i])[:,4]
            rodStress_i= prop_guard_helper.compute_rod_stress(nodePosHist_p[i])
            nodeMaxStressHist_p[i] = prop_guard_helper.compute_node_max_stress(rodStress_i, jointStress_i,crossJointStress_i)

        propGuardData[batchIdx*sampleNum+sampleIdx,0:3] = orientationSample[sampleIdx,0:3]
        propGuardData[batchIdx*sampleNum+sampleIdx,3:6] = orientationSample[sampleIdx,3:6]
        propGuardData[batchIdx*sampleNum+sampleIdx,6] = np.max(nodeMaxStressHist_p.flatten())

        # Find max stress in tensegrity
        file_t = open(folderName+"ten"+str(batchIdx)+"_"+str(sampleIdx)+".pickle", 'rb')
        data_t = pickle.load(file_t)
        file_t.close()
        Ps_t = data_t["P"]
        tHist_t = data_t["t"]
        YPR_t = data_t["attYPR"]

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
        
        tensegrityData[batchIdx*sampleNum+sampleIdx,0:3] = orientationSample[sampleIdx,0:3]
        tensegrityData[batchIdx*sampleNum+sampleIdx,3:6] = orientationSample[sampleIdx,3:6]
        tensegrityData[batchIdx*sampleNum+sampleIdx,6] = np.max(nodeMaxStressHist_t.flatten())

minStress = min([np.min(propGuardData[:,6]), np.min(propGuardData[:,6])])
maxStress = min([np.max(propGuardData[:,6]), np.max(propGuardData[:,6])])
normStress = plt.Normalize(minStress, maxStress)

print("meanPropGuardStress=", np.mean(propGuardData[:,6]))
print("meanTensegrity=", np.mean(tensegrityData[:,6]))
print("maxPropGuardStress=", np.max(propGuardData[:,6]))
print("maxTensegrity=", np.max(tensegrityData[:,6]))

"""
Store the analysis result as pickle file
"""
analysisResult=dict({"tensegrity":tensegrityData, "prop_guard":propGuardData}) 
file = open(saveFolderName+"result_"+"scale_"+str(scaleText)+".pickle", 'wb')
pickle.dump(analysisResult, file)
file.close()