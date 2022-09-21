# IMPORTS
from py3dmath import *
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from problem_setup import design_param
from tensegrity.design_tensegrity import tensegrity_design
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

class tensegrity_plotter():
    def __init__(self,design:tensegrity_design) -> None:
        self.design = design
        pass

    def draw_tensegrity(self,ax3d):
        """
        Draw the tensegrity
        """  
        for b,e in self.design.rods:
            coords = np.array([self.design.nodePos[b],self.design.nodePos[e]])
            ax3d.plot(coords[:,0], coords[:,1], coords[:,2], '-', linewidth=3, color=(3/255,107/255,251/255,0.8))
            
        for b,e in self.design.strings:
            coords = np.array([self.design.nodePos[b],self.design.nodePos[e]])
            ax3d.plot(coords[:,0], coords[:,1], coords[:,2], '--', linewidth=2, color=(251/255,17/255,19/255,0.8))

        for n in self.design.nodePos:
            ax3d.plot(n[0], n[1], n[2],'o', markersize = 8, color=(43/255,255/255,255/255,0.8))

        for i in range(4):
            p = self.design.propPos[i,:]
            q = p + np.array([0,0,1]) * 0.1 * self.design.rLPreSS
            ax3d.plot(p[0], p[1], p[2], 'bo')
            if (drawArrow):
                arrow = Arrow3D([p[0],q[0]],[p[1],q[1]],[p[2],q[2]], arrowstyle="->", color="purple", lw = 3, mutation_scale=25)
                ax3d.add_artist(arrow)
        return


fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')

ax3d.set_xlim3d([-0.2, 0.2])
ax3d.set_ylim3d([-0.2, 0.2])
ax3d.set_zlim3d([-0.2, 0.2])
ax3d.set_box_aspect(aspect = (2,2,2))
ax3d.grid(False)

param = design_param()
design = tensegrity_design(param)
design.design_from_propeller()
plotter = tensegrity_plotter(design)
plotter.draw_tensegrity(ax3d)
ax3d.view_init(-10, 15)
ax3d.dist = 5
plt.show()
