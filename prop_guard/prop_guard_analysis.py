import numpy as np
from py3dmath.py3dmath import *
from prop_guard.prop_guard_ode import *

"""
Helper functions to compute / record related variables of the tensegrity simulation 
"""

class prop_guard_analysis():
    def __init__(self,prop_guard_ode:prop_guard_ode) -> None:
        self.prop_guard_ode = prop_guard_ode
        self.nodeNum = prop_guard_ode.nodeNum
        self.dim = prop_guard_ode.dim
        self.rods = prop_guard_ode.rods
        self.dRodList = prop_guard_ode.dRodList

        self.kRodList = prop_guard_ode.kRodList
        self.baseLengthList = prop_guard_ode.baseLengthList
        self.massList =prop_guard_ode.massList
        
        self.joints = prop_guard_ode.joints
        self.kJointList =prop_guard_ode.kJointList
        self.kJointStressList =prop_guard_ode.kJointStressList
        self.dJointList =prop_guard_ode.dJointList
        
        self.crossJoints = prop_guard_ode.crossJoints
        self.kCrossJointList = prop_guard_ode.kCrossJointList
        self.dCrossJointList = prop_guard_ode.dCrossJointList
        self.kCrossJointStressList = prop_guard_ode.kCrossJointStressList
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

    def compute_cross_joint_angle_and_torque(self,nodes,vels):
        # compute joint angle, spring moment and damping moment at the joint
        crossJointInfo = np.zeros((len(self.crossJoints),5))
        for cjID in range(len(self.crossJoints)):
            crossJoint = self.crossJoints[cjID]
            n0 = nodes[crossJoint[0]]
            n1 = nodes[crossJoint[1]]
            n2 = nodes[crossJoint[2]]

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

            v01 = Vec3(vels[crossJoint[0]] - vels[crossJoint[1]])
            v0_tan_dir = -rotAxis012.cross(-e01) # when two rods are parallel, this vector is zero. Otherwise, it cross product of two uniform vector perpendicular to each other
            omega0 = v01.dot(v0_tan_dir)/l01 # angular rate due to rotation of 0-1 rod
            v21 = Vec3(vels[crossJoint[2]] - vels[crossJoint[1]])
            v2_tan_dir = rotAxis012.cross(e12) # positive tangential direction for node 2 to increase bending angle
            omega2 = v21.dot(v2_tan_dir)/l12 # angular rate due to rotation of 1-2 rod
            
            M_spring = (theta-np.pi/2)*self.kCrossJointList[cjID]
            M_damping = (omega0 + omega2)*self.dCrossJointList[cjID]

            crossJointInfo[cjID, 0] = theta 
            crossJointInfo[cjID, 1] = omega0+omega2 
            crossJointInfo[cjID, 2] = M_spring 
            crossJointInfo[cjID, 3] = M_damping
            # stress due to bending at the cross joint. It should always be positive, despite delta angle might be negative due to definition 
            crossJointInfo[cjID, 4] = np.abs((theta-np.pi/2))*self.kCrossJointStressList[cjID] 
        return crossJointInfo

    def compute_rod_force(self,nodePos):
        # compute all elastic forces in rods based on node position
        rodForce = np.zeros(len(self.rods))
        for i in range(len(self.rods)):
            c = self.rods[i]
            e = nodePos[c[0]] - nodePos[c[1]]
            l = np.linalg.norm(e) #current length
            rodForce[i] = self.kRodList[i]*(self.baseLengthList[i]-l) #[N]
        return rodForce

    def compute_rod_stress(self,nodePos):
        # Compute the stress of each rod
        return self.compute_rod_force(nodePos)/(self.prop_guard_ode.prop_guard.rA)

    def compute_node_max_stress(self,rodStress, jointStress, crossJointStress = None):
        # rodStress: len(rod) * 1, stress at each rod due to compression/extension 
        # jointStress: nodeNum * 1, stress at each node due to bending moment 
        # We compute the max stress each node is under. (Notice that stress on both sides of the node might be different)
        nodeMaxStress = np.zeros(self.nodeNum)
        for nodeID in range(self.nodeNum):
            rodStressList = [0] #Default 0 stress
            for i in range(len(self.rods)):
                if nodeID in self.rods[i]:
                    rodStressList.append(np.abs(rodStress[i]))

            jointStressList = [0] #Default zero stress
            for jointID in range(len(self.joints)):
                if nodeID == self.joints[jointID][1]:
                    jointStressList.append(np.abs(jointStress[jointID]))
            
        
            if not (crossJointStress is None):
                for crossjointID in range(len(self.crossJoints)):
                    if nodeID == self.crossJoints[crossjointID][1]:
                        jointStressList.append(np.abs(crossJointStress[crossjointID]))


            nodeMaxStress[nodeID] = max(rodStressList) + max(jointStressList)
        return nodeMaxStress

    def compute_joint_forces_for_record(self, nodes, vels, returnStress = False):
        """
        nodes: position vector of nodes 
        vels: velocity vector of nodes
        """
        rotationSpringF = np.zeros_like(nodes) # force on nodes that is equivalent to the effect of a rotational spring
        rotationDampingF = np.zeros_like(nodes) # damping force on nodes that stops rotation.
        if returnStress:
            jointStress = np.zeros(self.nodeNum)

        for jointID in range(len(self.joints)):
            joint = self.prop_guard.joints[jointID]
            n0 = nodes[joint[0]]
            n1 = nodes[joint[1]]
            n2 = nodes[joint[2]]

            l01 = np.linalg.norm(n1-n0)
            l12 = np.linalg.norm(n2-n1)
            e01 = Vec3(n1-n0)/l01
            e12 = Vec3(n2-n1)/l12
            
            cross012 = e01.cross(e12)
            norm012 = cross012.norm2()

            if norm012 < 1e-12:
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
            M_damping = omega0 + omega2
            
            F0_spring = ((M_spring/l01)*(-v0_tan_dir)).to_array().squeeze()
            F2_spring = ((M_spring/l12)*(-v2_tan_dir)).to_array().squeeze()
            
            F0_damping = ((M_damping/l01)*(-v0_tan_dir)).to_array().squeeze()
            F2_damping = ((M_damping/l12)*(-v2_tan_dir)).to_array().squeeze()

            rotationSpringF[joint[0]] += F0_spring
            rotationSpringF[joint[2]] += F2_spring
            rotationSpringF[joint[1]] += -(F0_spring+F2_spring)

            rotationDampingF[joint[0]] += F0_damping
            rotationDampingF[joint[2]] += F2_damping
            rotationDampingF[joint[1]] += -(F0_damping+F2_damping)
            jointStress[joint[1]] = theta*self.kJointStressList[jointID]

        if returnStress:
            return [rotationSpringF, rotationDampingF, jointStress]
        else:
            return [rotationSpringF, rotationDampingF]

    def compute_wall_collision_force(self, nodePos, nWall:Vec3, kWall, pWall:Vec3):
        # Compute the stress of each rod
        nodeF = np.zeros(self.nodeNum)
        nWall = nWall/nWall.norm2() #normalize the direction vector
        for i in range(self.nodeNum):
            d=(Vec3(nodePos[i])-pWall).dot(nWall)
            if d<0:
                normalForce = -kWall*d*nWall
                nodeF[i] = normalForce.norm2()
        return nodeF