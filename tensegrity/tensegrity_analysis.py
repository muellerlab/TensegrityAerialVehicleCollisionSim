import numpy as np
from py3dmath import *
from tensegrity.tensegrity_ode import *

"""
Helper functions to compute / record related variables of the rensegrity simulation 
"""

class tensegrity_analysis():
    def __init__(self,tensegrity_ode:tensegrity_ode) -> None:
        self.tensegrity_ode = tensegrity_ode
        self.massList =tensegrity_ode.massList
        self.nodeNum = tensegrity_ode.nodeNum
        self.dim = tensegrity_ode.dim
        self.faces = tensegrity_ode.tensegrity.faces

        # string
        self.dString = tensegrity_ode.dString
        self.kString = tensegrity_ode.kString
        self.sL0 =tensegrity_ode.sL0
        self.strings = tensegrity_ode.strings

        # rod
        self.numRod =tensegrity_ode.numRod
        self.rods = tensegrity_ode.rods
        self.kRodList = tensegrity_ode.kRodList
        self.dRodList = tensegrity_ode.dRodList
        self.rodLength0List = tensegrity_ode.rodLength0List

        # joints
        self.joints = tensegrity_ode.joints
        self.kJointList =tensegrity_ode.kJointList
        self.dJointList =tensegrity_ode.dJointList
        self.kJointStressList = tensegrity_ode.kJointStressList
        pass

    def compute_joint_angle_and_torque(self,nodes,vels):
        # compute joint angle, spring moment and damping moment at the joint
        jointInfo = np.zeros((len(self.joints),5))

        for jointID in range(len(self.joints)):
            joint = self.joints[jointID]
            n0 = nodes[joint[0]]
            n1 = nodes[joint[1]]
            n2 = nodes[joint[2]]

            l01 = np.linalg.norm(n1-n0)
            l12 = np.linalg.norm(n2-n1)
            e01 = Vec3(n1-n0)/l01
            e12 = Vec3(n2-n1)/l12
            
            cross012 = e01.cross(e12)
            norm012 = cross012.norm2()

            if norm012 < 1e-10:
                # If two rods are parallel. No rotation axis can be determined. 
                # We overwrite it with zero-vector so corresponding torque will be zero
                rotAxis012 = Vec3(0,0,0) 
            else:
                rotAxis012 = cross012/norm012
            cosTheta = e01.dot(e12)
            if cosTheta < -(1-1e-12):
                theta = np.pi
            elif cosTheta > (1-1e-12):
                theta = 0
            else:
                theta = np.arccos(cosTheta)

            v01 = Vec3(vels[joint[0]] - vels[joint[1]])
            v0_tan_dir = -rotAxis012.cross(-e01) # when two rods are parallel, this vector is zero. Otherwise, it cross product of two uniform vector perpendicular to each other
            omega0 = v01.dot(v0_tan_dir)/l01 # angular rate due to rotation of 0-1 rod
            v21 = Vec3(vels[joint[2]] - vels[joint[1]])
            v2_tan_dir = rotAxis012.cross(e12) # positive tangential direction for node 2 to increase bending angle
            omega2 = v21.dot(v2_tan_dir)/l12 # angular rate due to rotation of 1-2 rod
            
            M_spring = theta*self.kJointList[jointID]
            M_damping = (omega0 + omega2)*self.dJointList[jointID]

            jointInfo[jointID, 0] = theta 
            jointInfo[jointID, 1] = omega0+omega2 
            jointInfo[jointID, 2] = M_spring 
            jointInfo[jointID, 3] = M_damping
            jointInfo[jointID, 4] = theta*self.kJointStressList[jointID] # stress due to bending at the joint
        return jointInfo

    def compute_rod_force(self,nodePos):
        # compute all elastic forces in rods based on node position
        rodForce = np.zeros(self.numRod)
        for i in range(self.numRod):
            r = self.rods[i]
            e = nodePos[r[0]] - nodePos[r[1]]
            l = np.linalg.norm(e) #current length
            rodForce[i] = self.kRodList[i]*(self.rodLength0List[i]-l)
        return rodForce

    def compute_rod_stress(self,nodePos):
        return self.compute_rod_force(nodePos)/self.tensegrity_ode.tensegrity.rA

    def compute_string_force(self,nodePos):
        # compute all elastic forces in strings based on node position
        stringNum = len(self.strings)
        stringForce = np.zeros(stringNum)
        for i in range(stringNum):
            s = self.strings[i]
            e = nodePos[s[0]] - nodePos[s[1]]
            l = np.linalg.norm(e) #current length
            if l > self.sL0: #only tensile
                stringForce[i] = self.kString*(l-self.sL0)
        return stringForce

    def compute_node_max_stress(self,linkStress, jointStress):
        # linkStress: len(link) * 1, stress at each link due to compression/extension 
        # jointStress: nodeNum * 1, stress at each node due to bending moment 
        # We compute the max stress each node is under. (Notice that stress on both sides of the node might be different)
        nodeMaxStress = np.zeros(self.nodeNum)
        for nodeID in range(self.nodeNum):
            linkStressList = []
            for i in range(len(self.rods)):
                if nodeID in self.rods[i]:
                    linkStressList.append(np.abs(linkStress[i]))
            nodeMaxStress[nodeID] = np.max(linkStressList)

            for jointID in range(len(self.joints)):
                if nodeID == self.joints[jointID][1]:
                    nodeMaxStress[nodeID] += np.abs(jointStress[jointID])
        return nodeMaxStress

    def compute_wall_collision_force(self, nodePos, nWall:Vec3, kWall, pWall:Vec3):
        # Compute the stress of each link
        nodeF = np.zeros(self.nodeNum)
        nWall = nWall/nWall.norm2() #normalize the direction vector
        for i in range(self.nodeNum):
            d=(Vec3(nodePos[i])-pWall).dot(nWall)
            if d<0:
                normalForce = -kWall*d*nWall
                nodeF[i] = normalForce.norm2()
        return nodeF


    def compute_min_prop_dist_to_face(self, nodePos, propOffset = 0):
        """
        Warning: This tools involves geometric assumption that may no longer hold when large deformation takes place! 
        Please use this function with caution!

        Input: nodePos - position of tensegrity nodes and quadcopter nodes
               propOffset - vertical offset distance of propeller (i.e)

        Summary: For each quadcopter node, we find its two neighbors and create a plane with these nodes.
                 We assume the (quad node -> propeller) vector is vertical to this plane with a distance of given veritcal offset.
                 Then, we find the distance from propeller to a tensegrity face as the shortest distance from propeller point 
                 to the intersection between the propeller plane and the tensegrity face plane, which goes through the proller point and is parallel to the 3-node plane we found earlier. 

        Output: shortest distance to surface for each propeller. Notice that this distance should be smaller than propR to prevent exposure.         
        """
        # Compute the distance from propeller nodes to tensegrity surface

        quadNode = nodePos[12:,:]
        planeNodes = [[1,0,2],[3,1,0],[0,2,3],[2,3,1]] #Id of nodes that forms plane according to the right hand rule

        minDistList = np.zeros(4)
        COM = Vec3(np.mean(nodePos,axis=0))

        for propID in range(4):
            planeNids = planeNodes[propID]
            n0 = Vec3(quadNode[planeNids[0]])
            n1 = Vec3(quadNode[planeNids[1]])
            n2 = Vec3(quadNode[planeNids[2]])

            cn012 = (n2-n1).cross(n0-n1)
            v = cn012/cn012.norm2() # vertical direction 
            propPos = Vec3(quadNode[propID]) + v*propOffset
            minDist = 1e5
            for face in self.faces:
                m0 = Vec3(nodePos[face[0]])
                m1 = Vec3(nodePos[face[1]])
                m2 = Vec3(nodePos[face[2]])
                cm012 = (m2-m1).cross(m0-m1)
                nf = cm012/cm012.norm2() #normal vector of face 
                if (COM-m0).dot(nf)<0:
                    nf = -nf #COM is always "inside" 
                
                #Find distance to propeller center using geometry. 
                vSurfDist = np.abs((m0-propPos).dot(nf)) # verical distance to surface
                cosTheta = nf.dot(v) 
                if cosTheta == 0:
                    #Two normal vectors are vertical
                    surfDist = vSurfDist
                else:
                    theta = np.arccos(cosTheta) 
                    if theta > np.pi/2:
                        theta = np.pi-theta
                    surfDist = vSurfDist/np.sin(theta)
                if surfDist<minDist:
                    minDist = surfDist
            minDistList[propID] = minDist  
        return minDistList