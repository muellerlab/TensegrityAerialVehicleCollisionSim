# Tool to create 2D grid that corresponds to the uniform distribution on sphere surface 
# Reference: https://mathworld.wolfram.com/SpherePointPicking.html
import sys
import numpy as np
from py3dmath.py3dmath import *

class OrientDist():
    """
    Orientation uniform sampler used for collision simulation. 
    We randomly sample on sub-part of unit sphere surface (normally octant) of the body-fixed frame, 
    and represent the orientation before collision with the unit vector from center of unit sphere
    to the sampled point. At the corresponding collision orientation, the sampled vector is normal 
    to the wall (WLOG, we assume it is (-1,0,0))

    Sample orientation of the vehicle with the 
    """


    def __init__(self, thetaRange, phiRange, vCollision:Vec3, seed = 0) -> None:
        """
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
        thetaRange: [start, end] corresponds to the theta angle in spherical coordinate
        phiRange: [start, end] corresponds to the range of phi angle in spherical coordinate
        vCollision: collision vector, the direction we want the sampled vector to rotate to
        """
        self.thetaRange = thetaRange
        self.phiRange = phiRange
        self.u = [self.phiRange[0]/np.pi, self.phiRange[1]/np.pi]
        self.vCollision = vCollision 
        np.random.seed(seed)
        pass

    def uniform_sample(self, n):
        #each sample is in the form of [x,y,z,yaw,pitch,roll]  
        sample = np.zeros((n,6))
        for i in range(n):
            theta = np.random.uniform(self.thetaRange[0],self.thetaRange[1])
            phi = np.arccos(1-2*np.random.uniform(self.u[0],self.u[1]))
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            # Collision vector in body frame
            vB = Vec3([x,y,z])
            vB = vB/vB.norm2()
            
            crossB2E = vB.cross(self.vCollision)
            normB2E = crossB2E.norm2()

            if normB2E < 1e-12:
                # If two vectors parallel. No rotation between body frame and 
                rot = Rotation.identity()
            else:
                rotAxis = crossB2E/normB2E
                cosTheta = vB.dot(self.vCollision)
                if cosTheta < -(1-1e-12):
                    angle = np.pi
                elif cosTheta > (1-1e-12):
                    angle = 0
                else:
                    angle = np.arccos(cosTheta)
                rot = Rotation.from_axis_angle(rotAxis, angle)
                ypr = rot.to_euler_YPR()
            sample[i,:] = np.array([x,y,z,ypr[0],ypr[1],ypr[2]])

        return sample
    



