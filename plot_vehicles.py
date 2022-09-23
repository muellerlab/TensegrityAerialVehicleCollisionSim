# IMPORTS
from prop_guard.design_prop_guard import prop_guard_design
from py3dmath.py3dmath import *
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from problem_setup import design_param
from tensegrity.design_tensegrity import tensegrity_design
from prop_guard.design_prop_guard import prop_guard_design
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

drawArrow = False # If draw arrows that indicate the direction of thrust 

## https://stackoverflow.com/questions/38194247/how-can-i-connect-two-points-in-3d-scatter-plot-with-arrow
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

class design_plotter():
    def __init__(self, tensegrity:tensegrity_design, prop_guard:prop_guard_design = None) -> None:
        self.tensegrity = tensegrity
        self.prop_guard = prop_guard
        pass


    def draw_propeller_and_guard(self,ax, center, r, drawGuard = True, vehicleCenter = [0,0,0]):
        p = Circle((center[0], center[1]), r, color=(247/255,147/255,29/255,0.9)) 
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=center[2], zdir="z")

        if drawGuard:
            nodeNum = 8
            x = np.zeros(nodeNum)
            y = np.zeros(nodeNum)
            z = np.zeros(nodeNum)
            theta0 = np.arctan2(center[1]-vehicleCenter[1],center[0]-vehicleCenter[0])
            theta = np.linspace(theta0-np.pi/2, theta0+np.pi/2, nodeNum)

            p = Vec3([r*np.cos(theta), r*np.sin(theta),0*theta])
            x0 = p.x.squeeze() + center[0]
            y0 = p.y.squeeze() + center[1]
            z0 = p.z.squeeze() + center[2]
            ax.plot(x0, y0, z0 + r*0.2, linewidth=5, color=(95/255,138/255,125/255,0.8))
            ax.plot(x0, y0, z0 - r*0.2, linewidth=5, color=(95/255,138/255,125/255,0.8))

            for i in range(nodeNum):
                ax.plot([x0[i],x0[i]], [y0[i],y0[i]], [z0[i]-r*0.2, z0[i]+r*0.2], linewidth=5, color=(95/255,138/255,125/255,0.8))

    def draw_tensegrity(self,ax3d):
        """
        Draw the tensegrity
        """  
        nodePos = self.tensegrity.nodePos
        propPos = self.tensegrity.propPos
        for b,e in self.tensegrity.rods:
            coords = np.array([nodePos[b],nodePos[e]])
            ax3d.plot(coords[:,0], coords[:,1], coords[:,2], '-', linewidth=3, color=(3/255,107/255,251/255,0.8))
            
        for b,e in self.tensegrity.strings:
            coords = np.array([nodePos[b],nodePos[e]])
            ax3d.plot(coords[:,0], coords[:,1], coords[:,2], '--', linewidth=2, color=(251/255,17/255,19/255,0.8))

        for n in nodePos:
            ax3d.plot(n[0], n[1], n[2],'o', markersize = 8, color=(43/255,255/255,255/255,0.8))

        for i in range(6):
            p = propPos[i,:]
            q = p + np.array([0,0,1]) * 0.1 * self.tensegrity.rLPreSS
            ax3d.plot(p[0], p[1], p[2], 'bo')
            self.draw_propeller_and_guard(ax3d, p, self.tensegrity.propR, False)

            if (drawArrow):
                arrow = Arrow3D([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], arrowstyle="->", color="purple", lw = 3, mutation_scale=25)
                ax3d.add_artist(arrow)
            
        return

    def draw_propguard(self, ax3d, xshift = 0, yShift = 0, zShift=0):
        propRods = [[1,3],[6,7]]
        propNodes = np.zeros_like(self.prop_guard.nodePosList)
        propRot = Rotation.from_euler_YPR([np.pi/4,0,0])
        for i in range(self.prop_guard.nodeNum):
            propNodes[i] = (propRot*Vec3(self.prop_guard.nodePosList[i])).to_array().T

        for b,e in propRods:
            coords = np.array([propNodes[b],propNodes[e]])
            ax3d.plot(coords[:,0]+xshift, coords[:,1]+yShift, coords[:,2]+zShift, '-', linewidth=5, color=(3/255,107/255,251/255,0.8))

        propGuardPropNode = [1,3,6,7]
        for i in range(4):
            p = propNodes[propGuardPropNode[i],:]
            p[0] = p[0]+xshift
            p[1] = p[1]+yShift
            p[2] = p[2]+zShift
            ax3d.plot(p[0], p[1], p[2], 'bo')
            self.draw_propeller_and_guard(ax3d, p, self.prop_guard.param.propR, True, [xshift,yShift,zShift])

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')

ax3d.set_xlim3d([-0.4, 0.2])
ax3d.set_ylim3d([-0.1, 0.1])
ax3d.set_zlim3d([-0.1, 0.1])
ax3d.set_box_aspect(aspect = (6,2,2))
ax3d.grid(False)

param = design_param()
tensegrity = tensegrity_design(param)
tensegrity.design_from_propeller()

#prop_guard = prop_guard_design(param)
#prop_guard.get_pos_from_propeller()
#prop_guard.design()

#plotter = design_plotter(tensegrity,prop_guard)
plotter = design_plotter(tensegrity)

plotter.draw_tensegrity(ax3d)
#plotter.draw_propguard(ax3d,-0.2,0.0,0)
ax3d.view_init(-10, 15)
plt.axis('off')
ax3d.dist = 5
plt.show()
