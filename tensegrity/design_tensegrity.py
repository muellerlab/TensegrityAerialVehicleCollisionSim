import numpy as np
from py3dmath.py3dmath import *
from scipy.linalg import null_space
from problem_setup import design_param
from scipy.optimize import fsolve

class tensegrity_design():
    def __init__(self,param:design_param) -> None:
        self.dim = 3
        self.numTensegrityNode = 12
        self.numMassNode = 4
        self.nodeNum = self.numTensegrityNode + self.numMassNode

        self.numString = 24 
        self.numTensegrityRod = 6 
        self.numRod = self.numTensegrityRod + 4 #real rod in tensegrity + 4 rods due to additional connections of quadcopter mass nodes

        sM = param.mStructure/12 # mass of each structure node
        qM = param.mQuad/4 # mass of each quad node
        self.massList = np.array([sM for i in range(12)] + [qM for i in range(4)])
        
        self.param = param
        self.propR = param.propR

        # 12 tensegrity nodes + 4 mass nodes representing quadcopter weight
        unitTensegrityNodes = np.array([
            # Tensegrity node
            [0.00, 0.50, 0.25], 
            [1.00, 0.50, 0.25], 
            [0.00, 0.50, 0.75], 
            [1.00, 0.50, 0.75], 
            [0.25, 0.00, 0.50], 
            [0.25, 1.00, 0.50], 
            [0.75, 0.00, 0.50], 
            [0.75, 1.00, 0.50], 
            [0.50, 0.25, 0.00], 
            [0.50, 0.25, 1.00], 
            [0.50, 0.75, 0.00], 
            [0.50, 0.75, 1.00],
            ])
        COM = np.sum(unitTensegrityNodes[0:12,:],axis=0)/12
        self.unitNodePos = unitTensegrityNodes-COM # Move the unit node COM to [0,0,0]
        self.faces = self.get_faces(self.unitNodePos)
        self.strings = [
            [0, 4], 
            [0, 5], 
            [0, 8], 
            [0, 10], 
            [1, 6], 
            [1, 7], 
            [1, 8], 
            [1, 10], 
            [2, 4], 
            [2, 5], 
            [2, 9], 
            [2, 11], 
            [3, 6], 
            [3, 7], 
            [3, 9], 
            [3, 11], 
            [4, 8], 
            [4, 9], 
            [5, 10], 
            [5, 11], 
            [6, 8], 
            [6, 9], 
            [7, 10], 
            [7, 11]]

        # rods connecting tensegrity nodes
        self.fullRods = [
            [0, 1], 
            [2, 3], 
            [4, 5],
            [6, 7],
            [8, 9], 
            [10, 11]]

        # rod pieces connecting tensegrity nodes + quad mass nodes
        self.rods = [
            [0, 1], 
            [2, 3], 
            [4, 12],
            [12,13],
            [13, 5],
            [6, 14],
            [14, 15],
            [15, 7],
            [8, 9], 
            [10, 11]]

        # joints created by two neighboring rods connected 
        self.joints = [
            [4, 12, 13],
            [12, 13, 5],
            [6, 14, 15],
            [14, 15, 7]]     

        if self.param.dampingCase ==1:
            self.dRodList = self.param.dRod * np.ones(len(self.rods))
            self.dJointList = self.param.dJoint * np.ones(len(self.joints))
            self.dString = self.param.dString
        else:
            self.dRodList = np.zeros(len(self.rods))
            self.dJointList = np.zeros(len(self.joints))
        pass

    """
    General design tools
    """
    def findPretensionRatio(self,N, rods, strings):
        # Solve the equilibrium matrix to find the ratio between rod force and string force under pre-tension
        A = np.zeros((self.dim*self.numTensegrityNode, self.numTensegrityRod+self.numString)) # Each column is a bar and each (3 rows) is a node 
        strIDOffset = self.numTensegrityRod
        for i in range(self.numTensegrityRod):
            b = rods[i][0]
            e = rods[i][1]
            rodV = (N[b] - N[e])/np.linalg.norm(N[b] - N[e])
            A[3*b:3*(b+1),i] = rodV
            A[3*e:3*(e+1),i] = -rodV

        for j in range(self.numString):
            b = strings[j][0]
            e = strings[j][1]
            strV = (N[b] - N[e])/np.linalg.norm(N[b] - N[e])
            A[3*b:3*(b+1),j+strIDOffset] = -strV
            A[3*e:3*(e+1),j+strIDOffset] = strV

        nullSpace = null_space(A)
        ratio = np.abs(nullSpace[0]/nullSpace[strIDOffset])[0]
        return ratio
    

    def get_faces(self, unitNodes):
        # return enumerated list of faces, each face defined by the index of the three nodes making it up
        # each face is defined by three nodes:
        numNodes = len(unitNodes)
        
        # by design, n0 < n1 < n2
        # first, enumerate all possibilities
        faces = []
        for i0 in range(numNodes):
            for i1 in range(i0 + 1, numNodes):
                for i2 in range(i1 + 1, numNodes):
                    n0 = Vec3(unitNodes[i0])
                    n1 = Vec3(unitNodes[i1])
                    n2 = Vec3(unitNodes[i2])
                    # check if it makes a face:
                    c = (n0 + n1 + n2) / 3
                    if c.norm2() < 0.4:
                        continue # Only mid point of faces made with three points have distance larger than 0.4
                    faces += [[i0, i1, i2]] 
        return faces




    # Nonlinear equation system for the no-stress length and cross-sectional area of the string
    def funcString(self,x, sM, sRho, sLPreSS, sE, sPreT):
        # x[0] = sA, cross sectional area of string
        # x[1] = sL0, 0-strain string length 
        return [x[0]*x[1] - sM/(sRho*24), # mass equation
            sE*x[1]*x[0] + sPreT*x[1] - sE*sLPreSS*x[0]] #stress-strain relationship  

    # Nonlinear equation system for the no-stress length and cross-sectional area of the rod
    def funcRod(self,x, rM, rRho, rLPreSS, rE, rPreT):
        # x[0] = rA, cross sectional area of rod
        # x[1] = rL0, 0-strain rod length 
        return [x[0]*x[1] - rM/(rRho*6), # mass equation
            (x[1]-rLPreSS)*rE*x[0] - rPreT*x[1]] #stress-strain relationship  

    """
    Tool for design case 0: smallest + symmetric
    """
    def propDistConstrSymmetric(self,L,unitN0,unitN1,unitN2,unitNProp):
        # Constraint for the distance of the propeller mounting point to the tensegrity surface given 
        node = Vec3(unitNProp) #position of the propeller on the rod
        v0 = Vec3(unitN1-unitN0)
        v1 = Vec3(unitN2-unitN0)
        vP = v0.cross(v1)
        vP = vP/vP.norm2() #normalize 
        dist = (node-Vec3(unitN1)).dot(vP)
        return abs(dist)*L-self.propR

    def findMinimumRodLengthSymmetric(self, unitPropPos):
        # Quad mass node
        n0 = self.unitNodePos[4,:]
        n1 = self.unitNodePos[0,:]
        n2 = self.unitNodePos[2,:]
        rodLength = fsolve(self.propDistConstrSymmetric, np.array([0]), args=(n0, n1, n2, unitPropPos[0],self.propR))[0]
        return rodLength

    """
    Tool for design case 1: smallest + asymmetric
    """
    def propDistConstrAsymmetric(self, L, unitN0,unitN1,unitN2,unitPropX):
        # Constraint for the distance of the propeller mounting point to the tensegrity surface given 
        # N0, N1, N2 are three notes making up triangle surface. The propeller circle should be tanget to the 
        # face for the smallet tensegrity design. 
        node = Vec3(unitPropX*L,-self.propR,0) #position of the propeller on the rod  
        v0 = Vec3(unitN1-unitN0)
        v1 = Vec3(unitN2-unitN0)
        vP = v0.cross(v1)
        vP = vP/vP.norm2() #normalize 
        dist = (node-Vec3(unitN1)*L).dot(vP)
        return dist-self.propR

    def findMinimumRodLengthAsymmetric(self, unitPropX):
        # Quad mass node
        n0 = self.unitNodePos[4,:]
        n1 = self.unitNodePos[0,:]
        n2 = self.unitNodePos[2,:]
        rodLength = fsolve(self.propDistConstrAsymmetric, np.array([0.2]), args=(n0, n1, n2, unitPropX))[0]
        return rodLength
    
    """
    Design tensegrity based on propeller size. 
    Find the tensegrity with minimum size that can fully enclose the propeller. 
    """
    def design_from_propeller(self):
        stressRatio = self.findPretensionRatio(self.unitNodePos, self.fullRods, self.strings)

        # Assume the structure is made with six rods. Here we compute spec of each rod
        # We further assume that the change in cross sectional area due to deformation of pre-stress is negligible. 
        rM = self.param.mStructure*self.param.gamma_m/(1+self.param.gamma_m) # Total mass for rod 
        sM = self.param.mStructure/(1+self.param.gamma_m) # Total mass for string
        rV = (rM/self.param.rRho)/6 #[m^3] volume of each rod

        if self.param.designCase == 0:
            unitPropPos = np.array([
            [-0.25, -0.25, 0.0],
            [-0.25, 0.25, 0.0],
            [0.25, -0.25, 0.0],
            [0.25, 0.25, 0.0]])

            rLPreSS = self.findMinimumRodLengthSymmetric(unitPropPos)
            rPreT = self.param.sPreT * stressRatio # [N] rod pre-compression force

            root = fsolve(self.funcRod, [rM/(self.param.rRho*6*rLPreSS), rLPreSS], args=(rM, self.param.rRho, rLPreSS, self.param.rE, rPreT))
            self.rA = root[0] # cross sectional area
            rR = np.sqrt(self.rA/np.pi) #[m] diameter of rod
            self.rL = root[1] # no-stress string length
            rPreStrain = (self.rL-rLPreSS)/self.rL

            compressedPropPos = unitPropPos*rLPreSS
            compressedNodePos = np.vstack((rLPreSS*self.unitNodePos, compressedPropPos))

        elif self.param.designCase ==1:
            unitPropX = -0.25 # propeller is on one of the horizontal rod
            rLPreSS = self.findMinimumRodLengthAsymmetric(unitPropX)
            rPreT = self.param.sPreT * stressRatio # [N] rod pre-compression force
            
            root = fsolve(self.funcRod, [rM/(self.param.rRho*6*rLPreSS), rLPreSS], args=(rM, self.param.rRho, rLPreSS, self.param.rE, rPreT))
            self.rA = root[0] # cross sectional area
            self.rR = np.sqrt(self.rA/np.pi) #[m] diameter of rod
            self.rL = root[1] # no-stress string length
            rPreStrain = (self.rL-rLPreSS)/self.rL

            compressedPropPos = np.array([
            [-0.25*rLPreSS, -self.propR, 0.0],
            [-0.25*rLPreSS, self.propR, 0.0],
            [0.25*rLPreSS, -self.propR, 0.0],
            [0.25*rLPreSS, self.propR, 0.0]])
            compressedNodePos = np.vstack((rLPreSS*self.unitNodePos, compressedPropPos))

        self.rodLength0List = np.zeros(self.numRod) # list of rod length under zero stress
        self.kRodList = np.zeros(self.numRod) # list of stiffness of rods
        for i in range(self.numRod):
            b = self.rods[i][0]
            e = self.rods[i][1]
            self.rodLength0List[i] = np.linalg.norm(compressedNodePos[b] - compressedNodePos[e])/(1-rPreStrain)
            self.kRodList[i] = self.param.rE*self.rA/self.rodLength0List[i]
            if self.param.dampingCase == 0:
                eqMass = (self.massList[b] + self.massList[e])/2
                self.dRodList[i] =  2*np.sqrt(eqMass*self.kRodList[i])

        self.kJointList = np.zeros(len(self.joints))
        self.kJointStressList = np.zeros(len(self.joints))
        for i in range(len(self.joints)):
            [b,m,e] = self.joints[i]
            lbm = np.linalg.norm(compressedNodePos[b] - compressedNodePos[m])
            lme = np.linalg.norm(compressedNodePos[m] - compressedNodePos[e])
            jointLength = lbm + lme
            jointI = (self.rR**4)*np.pi/4  #Second moment of area 
            self.kJointList[i] = jointI*self.param.rE/(jointLength) # moment ~= k*theta, where theta is the bending angle. 
            self.kJointStressList[i] = self.kJointList[i]*self.rR/jointI # stress = Moment*r/I
            if self.param.dampingCase == 0:
                eqJ = (self.massList[b]*lbm**2 + self.massList[e]*lme**2)/2 #Equivalent mass moment of inertia
                self.dJointList[i] = 2*np.sqrt(self.kJointList[i]*eqJ)

        # length ratio between string and rod in a self-stressed tensegrity
        gamma_l = np.linalg.norm(self.unitNodePos[self.strings[0][0]]-self.unitNodePos[self.strings[0][1]])/np.linalg.norm(self.unitNodePos[self.fullRods[0][0]]-self.unitNodePos[self.fullRods[0][1]]) 
        sLPreSS = rLPreSS * gamma_l # length of string under pre-tension
        root = fsolve(self.funcString, [sM/(self.param.sRho*24*sLPreSS), sLPreSS], args=(sM, self.param.sRho, sLPreSS, self.param.sE, self.param.sPreT))
        sA = root[0] # cross sectional area
        self.sL0 = root[1] # no-stress string length

        self.kString = self.param.sE*sA/self.sL0
        if self.param.dampingCase == 0:
            #Notice that all strings are identical
            b = self.strings[0][0]
            e = self.strings[0][1]
            eqMass = (self.massList[b] + self.massList[e])/2
            self.dString =  2*np.sqrt(eqMass*self.kString)

        self.nodePos = compressedNodePos 
        self.propPos = compressedPropPos
        self.rLPreSS = rLPreSS
        self.sLPreSS = sLPreSS
    

    """
    Design tensegrity based on rod size used to design tensegrity.
    Indicate propeller's position with propOffSet parameter
    rOR: outter radius. 
    rIR: inner radius, zero for solid rod, non zero for tubes. 
    """
    def design_from_rod(self, rL0, rOR, propOffSet, sA, rIR = 0):
        stressRatio = self.findPretensionRatio(self.unitNodePos, self.fullRods, self.strings)
        self.rL = rL0
        self.rA = np.pi*(rOR**2-rIR**2)
        rPreT = self.param.sPreT * stressRatio # [N] rod pre-compression force
        rPreStrain = (rPreT/self.rA)/self.param.rE
        rLPreSS = self.rL*(1-rPreStrain)
        self.rR = rOR

        compressedPropPos = np.array([
        [-0.25*rLPreSS, -propOffSet, 0.0],
        [-0.25*rLPreSS, propOffSet, 0.0],
        [0.25*rLPreSS, -propOffSet, 0.0],
        [0.25*rLPreSS, propOffSet, 0.0]])
        compressedNodePos = np.vstack((rLPreSS*self.unitNodePos, compressedPropPos))

        self.nodePos = compressedNodePos 
        self.propPos = compressedPropPos

        self.rodLength0List = np.zeros(self.numRod) # list of rod length under zero stress
        self.kRodList = np.zeros(self.numRod) # list of stiffness of rods
        for i in range(self.numRod):
            b = self.rods[i][0]
            e = self.rods[i][1]
            self.rodLength0List[i] = np.linalg.norm(compressedNodePos[b] - compressedNodePos[e])/(1-rPreStrain)
            self.kRodList[i] = self.param.rE*self.rA/self.rodLength0List[i]
            if self.param.dampingCase == 0:
                eqMass = (self.massList[b] + self.massList[e])/2
                self.dRodList[i] =  2*np.sqrt(eqMass*self.kRodList[i])

        self.kJointList = np.zeros(len(self.joints))
        self.kJointStressList = np.zeros(len(self.joints))
        for i in range(len(self.joints)):
            [b,m,e] = self.joints[i]
            lbm = np.linalg.norm(compressedNodePos[b] - compressedNodePos[m])
            lme = np.linalg.norm(compressedNodePos[m] - compressedNodePos[e])
            jointLength = lbm + lme
            jointI = (self.rR**4)*np.pi/4  #Second moment of area 
            self.kJointList[i] = jointI*self.param.rE/(jointLength) # moment ~= k*theta, where theta is the bending angle. 
            self.kJointStressList[i] = self.kJointList[i]*self.rR/jointI # stress = Moment*r/I
            if self.param.dampingCase == 0:
                eqJ = (self.massList[b]*lbm**2 + self.massList[e]*lme**2)/2 #Equivalent mass moment of inertia
                self.dJointList[i] = 2*np.sqrt(self.kJointList[i]*eqJ)

        # length ratio between string and rod in a self-stressed tensegrity
        gamma_l = np.linalg.norm(self.unitNodePos[self.strings[0][0]]-self.unitNodePos[self.strings[0][1]])/np.linalg.norm(self.unitNodePos[self.fullRods[0][0]]-self.unitNodePos[self.fullRods[0][1]]) 
        self.sLPreSS = rLPreSS * gamma_l # length of string under pre-tension
        sPreStrain = (self.param.sPreT /sA)/self.param.sE
        self.sL0 = self.sLPreSS/(1+sPreStrain)
        self.rLPreSS = rLPreSS
        
        self.kString = self.param.sE*sA/self.sL0
        if self.param.dampingCase == 0:
            #Notice that all strings are identical
            b = self.strings[0][0]
            e = self.strings[0][1]
            eqMass = (self.massList[b] + self.massList[e])/2
            self.dString =  2*np.sqrt(eqMass*self.kString)

        return 