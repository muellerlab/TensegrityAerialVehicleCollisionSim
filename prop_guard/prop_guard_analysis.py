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
        self.links = prop_guard_ode.links
        self.dRod = prop_guard_ode.dRod
        self.kLinkList = prop_guard_ode.kLinkList
        self.baseLengthList = prop_guard_ode.baseLengthList
        self.massList =prop_guard_ode.massList
        self.joints = prop_guard_ode.joints
        self.kJointList =prop_guard_ode.kJointList
        self.kJointStressList =prop_guard_ode.kJointStressList
        self.dJoint =prop_guard_ode.dJoint
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
            M_damping = (omega0 + omega2)*self.dJoint

            jointInfo[jointID, 0] = theta 
            jointInfo[jointID, 1] = omega0+omega2 
            jointInfo[jointID, 2] = M_spring 
            jointInfo[jointID, 3] = M_damping
            jointInfo[jointID, 4] = theta*self.kJointStressList[jointID] # stress due to bending at the joint
        return jointInfo

    def compute_link_force(self,nodePos):
        # compute all elastic forces in rods based on node position
        linkForce = np.zeros(len(self.links))
        for i in range(len(self.links)):
            c = self.links[i]
            e = nodePos[c[0]] - nodePos[c[1]]
            l = np.linalg.norm(e) #current length
            linkForce[i] = self.kLinkList[i]*(self.baseLengthList[i]-l) #[N]
        return linkForce

    def compute_link_stress(self,nodePos):
        # Compute the stress of each link
        return self.compute_link_force(nodePos)/(self.prop_guard_ode.prop_guard.rA)

    def compute_node_max_stress(self,linkStress, jointStress):
        # linkStress: len(link) * 1, stress at each link due to compression/extension 
        # jointStress: nodeNum * 1, stress at each node due to bending moment 
        # We compute the max stress each node is under. (Notice that stress on both sides of the node might be different)
        nodeMaxStress = np.zeros(self.nodeNum)
        for nodeID in range(self.nodeNum):
            linkStressList = []
            for i in range(len(self.links)):
                if nodeID in self.links[i]:
                    linkStressList.append(np.abs(linkStress[i]))
            nodeMaxStress[nodeID] = np.max(linkStressList)

            for jointID in range(len(self.joints)):
                if nodeID == self.joints[jointID][1]:
                    nodeMaxStress[nodeID] += np.abs(jointStress[jointID])
        return nodeMaxStress

    def compute_joint_forces_for_record(self,nodes, vels, returnStress = False):
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