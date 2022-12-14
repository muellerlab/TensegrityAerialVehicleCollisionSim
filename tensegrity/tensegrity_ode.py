import numpy as np
from py3dmath.py3dmath import *
from tensegrity.design_tensegrity import tensegrity_design

"""
Functions for dynamics simulation
"""

class tensegrity_ode():
    def __init__(self, tensegrity:tensegrity_design) -> None:
        self.tensegrity = tensegrity
        self.nodeNum = tensegrity.nodeNum
        self.dim = tensegrity.dim
        self.massList =tensegrity.massList
        self.rL = tensegrity.rL 

        # string
        self.dString = tensegrity.dString
        self.kString = tensegrity.kString
        self.sL0 =tensegrity.sL0
        self.strings = tensegrity.strings

        # rod
        self.numRod =tensegrity.numRod
        self.rods = tensegrity.rods
        self.kRodList = tensegrity.kRodList
        self.dRodList = tensegrity.dRodList
        self.rodLength0List = tensegrity.rodLength0List

        # joints
        self.joints = tensegrity.joints
        self.kJointList =tensegrity.kJointList
        self.dJointList =tensegrity.dJointList
        self.kJointStressList = tensegrity.kJointStressList
        pass


    def compute_joint_forces(self, nodes, vels, jointID):
        """
        nodes: position vector of nodes 
        vels: velocity vector of nodes
        joint: ID of the three nodes that make up the joint. 
        """
        joint = self.tensegrity.joints[jointID]
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
        M_damping = (omega0 + omega2)*self.dJointList[jointID]

        F0 = (((M_spring+M_damping)/l01)*-v0_tan_dir).to_array().squeeze()
        F2 = (((M_spring+M_damping)/l12)*-v2_tan_dir).to_array().squeeze()
        F1 = -(F0 + F2) 
        return [F0, F1, F2]

    def compute_internal_forces(self, P, sepReturn = False):
        # Compute the internal elastic and damping forces of the tensegrity system 
        # When sepReturn flag is true, return elastic and damping force seperately

        nodes = P[:self.nodeNum*self.dim].reshape((self.nodeNum,self.dim)) # num nodes x  3D 
        vels = P[self.nodeNum*self.dim:].reshape((self.nodeNum,self.dim))  

        elasticF = np.zeros_like(nodes)
        dampingF = np.zeros_like(vels)
        rotF = np.zeros_like(nodes) #force related to relative rotation of rods

        for node in range(self.nodeNum):
            for rodID in range(self.numRod):
                r = self.rods[rodID]
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
                    dampingF[node] += -self.dRodList[rodID]*(vab.dot(e/l))*(e/l) 
                    elasticF[node] += self.kRodList[rodID]*(l-self.rodLength0List[rodID])*e/l

            for s in self.strings:
                if node in s:
                    # direction of cable:
                    e = nodes[s[0]] - nodes[s[1]]
                    if s[0] == node:
                        e = -e #get the sign right
                        vaE = vels[s[0]]
                        vbE = vels[s[1]]
                    else:
                        vaE = vels[s[1]]
                        vbE = vels[s[0]]
                    l = np.linalg.norm(e) #current length
                    vab = vaE - vbE

                    # Bidrectional damping:
                    dampingF[node] += -self.dString*(vab.dot(e/l))*(e/l) 
                    
                    # Unidirectional damping
                    # dampingM = -stringDamping*(vab.dot(e/l)) #projection of damping along string direction
                    # if dampingM >=0:
                    #     dampingF[node] += dampingM*(e/l) #damping can only pull but not push 
                    
                    if l > self.sL0:  # only tensile
                        elasticF[node] += self.kString*(l-self.sL0)*e/l

            for jointID in range(len(self.joints)):
                [F0, F1, F2] =  self.compute_joint_forces(nodes,vels,jointID)
                rotF[self.joints[jointID][0]] += F0
                rotF[self.joints[jointID][1]] += F1
                rotF[self.joints[jointID][2]] += F2
        if sepReturn:
            return [elasticF, dampingF, rotF]
        else:
            return elasticF + dampingF + rotF
            

    def ode_ivp_hit_force(self, t, P, extF, ft):
        dPdt = np.zeros_like(P)
        dPdt[:self.nodeNum*self.dim] = P[self.nodeNum*self.dim:]
        forces = self.compute_internal_forces(P)
        if t <= ft:
            forces += extF 
        for i in range(self.nodeNum):
            dPdt[self.nodeNum*self.dim+self.dim*i:self.nodeNum*self.dim+self.dim*(i+1)] = forces[i]/self.massList[i]
        return dPdt 

    def ode_ivp(self, t, P):
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
        d4 = Vec3(nodes[5]-nodes[4])/np.linalg.norm(nodes[5]-nodes[4])
        f4 = (totalF/2)*(rot*d4)
        d6 = Vec3(nodes[7]-nodes[6])/np.linalg.norm(nodes[7]-nodes[6])
        f6 = (totalF/2)*(rot*(d6/d6.norm2()))

        forces[4] += f4.to_array().squeeze()
        forces[6] += f6.to_array().squeeze()
        
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
    
    def eventAttr():
        def decorator(func):
            func.direction = 1
            func.terminal = True
            return func
        return decorator


    @eventAttr()
    def wall_check(self, t, P, nWall:Vec3, kWall, pWall:Vec3):
        """
        Check if the structure has stopped touching the wall. 
        Here we compute the smallest distance from structure to the wall
        """
        dMin = -self.rL
        nodes = P[:self.nodeNum*self.dim].reshape((self.nodeNum,self.dim))  
        for i in range(self.nodeNum):
            d=(Vec3(nodes[i])-pWall).dot(nWall)
            if d>dMin:
                dMin = d
        return dMin

    @eventAttr()
    def wall_check_simple(self, t, P, nWall:Vec3, kWall, pWall:Vec3):
        """
        Check if the structure has stopped touching the wall. 
        To decrease the computation. We assume: 1) normal direction of the wall is +x direction
                                                2) the wall surface is at the plane that x=0
        """
        nodes = P[:self.nodeNum*self.dim].reshape((self.nodeNum,self.dim))
        dMin = np.min(nodes[:,0])        
        return dMin-self.rL/20.0 #ends when the closest point to wall is 1/20 rL from the surface
    
    @eventAttr()
    def vel_check_simple(self, t, P, nWall:Vec3, kWall, pWall:Vec3):
        """
        Stop the simulation when COM velocity is pointing away from the wall with a certain threshold speed.
        We expect maximum deformation to take place before this moment.  
        """
        threshold = 0.1 #[m/s]
        vels = P[self.nodeNum*self.dim:].reshape((self.nodeNum,self.dim)) 
        vel_x = vels[:,0]
        return (vel_x-threshold).dot(self.massList) #weighted average of node velocity