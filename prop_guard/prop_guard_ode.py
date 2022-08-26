from typing import List
import numpy as np
from prop_guard.design_prop_guard import prop_guard_design
from py3dmath.py3dmath import *

"""
Functions to compute forces within the propeller guard 
and the ODE used to describe the dynamics of the system.
"""

class prop_guard_ode():
    def __init__(self, prop_guard:prop_guard_design) -> None:
        self.prop_guard = prop_guard
        self.nodeNum = prop_guard.nodeNum
        self.dim = prop_guard.dim
        self.links = prop_guard.links
        self.dRod = prop_guard.dRod
        self.kLinkList = prop_guard.kLinkList

        self.joints = prop_guard.joints
        self.kJointList =prop_guard.kJointList
        self.dJoint =prop_guard.dJoint
        self.kJointStressList = prop_guard.kJointStressList

        self.baseLengthList = prop_guard.baseLengthList
        self.massList =prop_guard.massList
        pass

    def compute_joint_forces(self, nodes, vels, jointID):
        """
        nodes: position vector of nodes 
        vels: velocity vector of nodes
        joint: ID of the three nodes that make up the joint. 
        """
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
        M_damping = (omega0 + omega2)*self.dJoint

        F0 = (((M_spring+M_damping)/l01)*-v0_tan_dir).to_array().squeeze()
        F2 = (((M_spring+M_damping)/l12)*-v2_tan_dir).to_array().squeeze()
        F1 = -(F0 + F2) 
        return [F0, F1, F2]


    def compute_internal_forces(self,P, sepReturn = False):
        # Compute the internal elastic and damping forces of the prop guard
        nodes = P[:self.nodeNum*self.dim].reshape((self.nodeNum,self.dim)) # 9 nodes x  3D 
        vels = P[self.nodeNum*self.dim:].reshape((self.nodeNum,self.dim))  
        
        springF = np.zeros_like(nodes) # linear spring force   
        dampF = np.zeros_like(nodes) # linear damping force
        rotF = np.zeros_like(nodes) # spring and damping force due to the relative rotaiton of springs 
        
        for node in range(self.nodeNum):
            for i in range(len(self.links)):
                r = self.links[i]
                if node in r:
                    # direction of rod:
                    e = nodes[r[0]] - nodes[r[1]]
                    if r[0] == node:
                        e = -e  # get the sign right
                        vaE = vels[r[0]] # earth frame velocity of node in current iteration
                        vbE = vels[r[1]] 
                    else:
                        vaE = vels[r[1]]
                        vbE = vels[r[0]]
                    l = np.linalg.norm(e)  # current length
                    vab = vaE - vbE
                    dampF[node] += -self.dRod*(vab.dot(e/l))*(e/l) 
                    springF[node] += self.kLinkList[i] * (l-self.baseLengthList[i])*e/l

        for jointID in range(len(self.joints)):
            [F0, F1, F2] =  self.compute_joint_forces(nodes,vels,jointID)
            rotF[self.joints[jointID][0]] += F0
            rotF[self.joints[jointID][1]] += F1
            rotF[self.joints[jointID][2]] += F2
        if sepReturn:
            return [springF, dampF, rotF]
        else:
            return springF + dampF + rotF

    def ode_ivp_hit_force(self,t, P, extF, ft):
        dPdt = np.zeros_like(P)
        dPdt[:self.nodeNum*self.dim] = P[self.nodeNum*self.dim:]
        forces = self.compute_internal_forces(P)

        if t<= ft:
            forces += extF 
        for i in range(self.nodeNum):
            dPdt[self.nodeNum*self.dim+self.dim*i:self.nodeNum*self.dim+self.dim*(i+1)] = forces[i]/self.massList[i]
        return dPdt 

    def ode_ivp(self,t, P):
        dPdt = np.zeros_like(P)
        dPdt[:self.nodeNum*self.dim] = P[self.nodeNum*self.dim:]
        forces = self.compute_internal_forces(P)
        for i in range(self.nodeNum):
            dPdt[self.nodeNum*self.dim+self.dim*i:self.nodeNum*self.dim+self.dim*(i+1)] = forces[i]/self.massList[i]
        return dPdt 

    def ode_ivp_force_body_frame(self, t, P, totalF, rot:Rotation):
        dPdt = np.zeros_like(P)
        dPdt[:self.nodeNum*self.dim] = P[self.nodeNum*self.dim:]
        forces = self.compute_internal_forces(P)

        nodes = P[:self.nodeNum*self.dim].reshape((self.nodeNum,self.dim)) 
        d0 = Vec3(nodes[5]-nodes[0])/np.linalg.norm(nodes[5]-nodes[0])
        f0 = (totalF/2)*(rot*d0)
        d8 = Vec3(nodes[4]-nodes[8])/np.linalg.norm(nodes[4]-nodes[8])
        f8 = (totalF/2)*(rot*d8)
        
        forces[0] += f0.to_array().squeeze()
        forces[8] += f8.to_array().squeeze()
        for i in range(self.nodeNum):
            dPdt[self.nodeNum*self.dim+self.dim*i:self.nodeNum*self.dim+self.dim*(i+1)] = forces[i]/self.massList[i]
        return dPdt 

    

    def ode_ivp_wall(self, t, P, nWall:Vec3, kWall, pWall:Vec3):
        """
        nWall: a normal vector pointing out of the wal
        pWall: a point on the wall surface
        kWall: Hooke's constant of the wall
        wallX: x coordinate of the wall surface. Any point with pos.x < wallX is in the wall
        kWall: Hooke's coefficient of the wall 

        Assume the wall is a very stiff spring.
        """
        dPdt = np.zeros_like(P)
        dPdt[:self.nodeNum*self.dim] = P[self.nodeNum*self.dim:]
        forces = self.compute_internal_forces(P)
        nodes = P[:self.nodeNum*self.dim].reshape((self.nodeNum,self.dim)) 

        nWall = nWall/nWall.norm2() #normalize the direction vector
        for i in range(self.nodeNum):
            d=(Vec3(nodes[i])-pWall).dot(nWall)
            if d<0:
                normalForce = -kWall*d*nWall
                forces[i] += normalForce.to_array().squeeze()
        for i in range(self.nodeNum):
            dPdt[self.nodeNum*self.dim+self.dim*i:self.nodeNum*self.dim+self.dim*(i+1)] = forces[i]/self.massList[i]
        return dPdt 