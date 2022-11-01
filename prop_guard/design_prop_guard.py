from traceback import print_tb
import numpy as np
from py3dmath import *
from scipy.linalg import null_space
from problem_setup import design_param
from tensegrity.design_tensegrity import tensegrity_design
# The Flight Plate 

"""
Two different design cases: 

0: assume tensegrity vehicle is symmetric, find the shortest rod length that can enclose it. 
Meanwhile, the prop guard protects the same vehicle that is hosted on the tensegrity

1: Ignore the tensegrity vehicle. Given propeller size, assume the vehicle with propguard is symmetric, find the smallest frame that can host it. 
"""

"""
Side:
     ∞         ∞
     |         |   
[----O---------O----]
n0  n1        n2   n3

Top view: 

        '---'8
          O  7
          |
          |
[----O----o 2----O----]
0    1    |      3    4
          |
          O  6
        '---'5

^Y
|
----> X
"""
class prop_guard_design():
    def __init__(self,param:design_param) -> None:
        self.dim = 3 # Simulation is in 3D space
        self.nodeNum = 9 # 9 nodes in total
        self.param = param
        sM = self.param.mStructure/5 
        qM = self.param.mQuad/4
        self.massList = np.array([sM, qM, sM, qM, sM, sM, qM, qM, sM])

        # The rod is defined as a rod piece connecting two nodes. 
        # Each rod is modelled as a linear spring-damper pair 
        self.rods = [
            [0, 1], 
            [1, 2], 
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 2],
            [2, 7],
            [7, 8]]

        # Assume every two neighboring rods are connected with a torque spring-damper pair
        self.joints = [
            [0,1,2],
            [1,2,3],
            [2,3,4],
            [5,6,2],
            [6,2,7],
            [2,7,8]]

        self.crossJoints = [[6,2,1],
                           [3,2,6],
                           [7,2,3],
                           [1,2,7]] #4 cross joints

        if self.param.dampingCase ==1:
            self.dRodList = self.param.dRod * np.ones(len(self.rods))
            self.dJointList = self.param.dJoint * np.ones(len(self.joints))
            self.dCrossJointList = self.param.dJoint * np.ones(len(self.crossJoints))
        else:
            self.dRodList = np.zeros(len(self.rods))
            self.dJointList = np.zeros(len(self.joints))
            self.dCrossJointList = np.zeros(len(self.crossJoints))
        pass
    
    def get_pos_from_comparable_tensegrity(self,propPos):
        propR = self.param.propR
        self.nodePosList[1,:] = propPos[1,:]
        self.nodePosList[7,:] = propPos[3,:]
        self.nodePosList[3,:] = propPos[2,:]
        self.nodePosList[6,:] = propPos[0,:]
        self.nodePosList[0,:] = self.nodePosList[1,:]*(propR+np.linalg.norm(self.nodePosList[1,:]))/(np.linalg.norm(self.nodePosList[1,:]))
        self.nodePosList[8,:] = self.nodePosList[7,:]*(propR+np.linalg.norm(self.nodePosList[7,:]))/(np.linalg.norm(self.nodePosList[7,:]))
        self.nodePosList[4,:] = self.nodePosList[3,:]*(propR+np.linalg.norm(self.nodePosList[3,:]))/(np.linalg.norm(self.nodePosList[3,:]))
        self.nodePosList[5,:] = self.nodePosList[6,:]*(propR+np.linalg.norm(self.nodePosList[6,:]))/(np.linalg.norm(self.nodePosList[6,:]))
        return
    
    def get_pos_from_propeller(self):
        propR = self.param.propR
        armLength = self.param.propR*np.sqrt(2)
        self.nodePosList = np.zeros((self.nodeNum,self.dim))
        self.nodePosList[1,:] = np.array([-armLength,0,0])
        self.nodePosList[7,:] = np.array([0,armLength,0])
        self.nodePosList[3,:] = np.array([armLength,0,0])
        self.nodePosList[6,:] = np.array([0,-armLength,0])

        self.nodePosList[0,:] = np.array([-(armLength+propR),0,0])
        self.nodePosList[8,:] = np.array([0,(armLength+propR),0])
        self.nodePosList[4,:] = np.array([(armLength+propR),0,0])
        self.nodePosList[5,:] = np.array([0,-(armLength+propR),0])

    def design(self,propPos = None):
        if self.param.designCase ==0:
            self.get_pos_from_comparable_tensegrity(propPos)
        else:
            self.get_pos_from_propeller()
        
        rV = (self.param.mStructure/self.param.rRho)/2 #Assume the frame is made with two rods. 
        self.rL = np.linalg.norm(self.nodePosList[4,:]-self.nodePosList[0,:])
        self.rA = rV/self.rL
        self.rR = np.sqrt(self.rA/np.pi) 

        self.rodLength = np.zeros(len(self.rods))
        self.kRodList = np.zeros(len(self.rods))
        for i in range(len(self.rods)):
            [b,e] = self.rods[i]
            self.rodLength[i] = np.linalg.norm(self.nodePosList[b] - self.nodePosList[e])
            self.kRodList[i] = self.param.rE*self.rA/(self.rodLength[i])

            if self.param.dampingCase == 0:
                eqMass = (self.massList[b] + self.massList[e])/2
                self.dRodList[i] = 2*np.sqrt(self.kRodList[i]*eqMass)

        self.baseLengthList = self.rodLength # list of each length components
        self.kJointList = np.zeros(len(self.joints))
        self.kJointStressList = np.zeros(len(self.joints)) # Coefficient for max stress at the joint node. 
        for i in range(len(self.joints)):
            [b,m,e] = self.joints[i]
            lbm = np.linalg.norm(self.nodePosList[b] - self.nodePosList[m])
            lme = np.linalg.norm(self.nodePosList[m] - self.nodePosList[e])
            jointLength = lbm + lme
            jointI = (self.rR**4)*np.pi/4  #Second moment of area 
            self.kJointList[i] = jointI*self.param.rE/(jointLength) # moment ~= k*theta, where theta is the bending angle. 
            self.kJointStressList[i] = self.kJointList[i]*self.rR/jointI # stress = Moment*r/I
            if self.param.dampingCase == 0:
                eqJ = (self.massList[b]*lbm**2 + self.massList[e]*lme**2)/2 #Equivalent mass moment of inertia
                self.dJointList[i] = 2*np.sqrt(self.kJointList[i]*eqJ)

        self.kCrossJointList = np.zeros(len(self.crossJoints))
        self.kCrossJointStressList = np.zeros(len(self.crossJoints)) # Coefficient for max stress at the crossjoint node. 
        for i in range(len(self.crossJoints)):
            [b,m,e] = self.crossJoints[i]
            lbm = np.linalg.norm(self.nodePosList[b] - self.nodePosList[m])
            lme = np.linalg.norm(self.nodePosList[m] - self.nodePosList[e])
            crossJointLength = lbm + lme            
            crossJointI = (self.rR**4)*np.pi/4  #Second moment of area 
            self.kCrossJointList[i] = crossJointI*self.param.rE/(crossJointLength) # moment ~= k*theta, where theta is the bending angle. 
            self.kCrossJointStressList[i] = self.kCrossJointList[i]*self.rR/crossJointI # stress = Moment*r/I
            if self.param.dampingCase == 0:
                eqJ = (self.massList[b]*lbm**2 + self.massList[e]*lme**2)/2 #Equivalent mass moment of inertia
                self.dCrossJointList[i] = 2*np.sqrt(self.kJointList[i]*eqJ)