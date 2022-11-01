import numpy as np
import scipy as sp
from py3dmath.py3dmath import Rotation, Vec3
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import proj3d,Axes3D
import matplotlib.colors as colors
import seaborn as sns
from problem_setup import design_param
import seaborn as sns
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator,Rbf,interp1d

"""
Helpful tools for simulation data visulazation
"""

"""
Tools to create truncate color scheme
"""
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmapReds = plt.get_cmap('Reds')
truncateReds = truncate_colormap(cmapReds, 0.5, 0.9) #Avoid the white part in the map
cmapGreens = plt.get_cmap('Greens')
truncateGreens = truncate_colormap(cmapGreens, 0.5, 0.9) #Avoid the white part in the map
cmapBlues = plt.get_cmap('Blues')
truncateBlues = truncate_colormap(cmapBlues, 0.5, 0.9) #Avoid the white part in the map



class Arrow3D(FancyArrowPatch):
    ### https://stackoverflow.com/questions/38194247/how-can-i-connect-two-points-in-3d-scatter-plot-with-arrow
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

"""
Extrapolate color map if it is not in the scatter range
https://stackoverflow.com/questions/20516762/extrapolate-with-linearndinterpolator
"""
class LinearNDInterpolatorExt(object):
    def __init__(self, points, values):
        self.funcinterp = LinearNDInterpolator(points, values)
        self.funcnearest = NearestNDInterpolator(points, values)
    
    def __call__(self, *args):
        z = self.funcinterp(*args)
        chk = np.isnan(z)
        if chk.any():
            return np.where(chk, self.funcnearest(*args), z)
        else:
            return z


class Octant_Mapping():
    """
    Tool to draw heat map and contour plot on a 3d octant surface
    """

    def __init__(self):
        return 

    def octantContour(self, ax, contour):
        for item in contour.collections:
            for i in item.get_paths():
                v = i.vertices
                theta_ct = v[:, 0]
                phi_ct = v[:, 1]
                # Map the vertex on 2D contour to 3D 
                x_ct = np.sin(phi_ct) * np.cos(theta_ct)
                y_ct = np.sin(phi_ct) * np.sin(theta_ct)
                z_ct = np.cos(phi_ct)
                ax.plot(x_ct,y_ct,z_ct,'k')

def map_colors(p3dc, data, cmap='viridis'):
    """
    Color a tri-mesh according according to the nodes. 
    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    data: the original 4d data [x,y,z,value]
    cmap: a colormap NAME, as a string
    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """
    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = np.array([np.array((x[s],y[s],z[s])).T for s in slices])
    values = np.zeros(triangles.shape[0])

    for i in range(triangles.shape[0]):
        tri = triangles[i]
        #define the triangle value as the average of three nodes
        id1 = np.where((data[:,0:3] == tri[0,:]).all(axis=1))[0]
        id2 = np.where((data[:,0:3] == tri[1,:]).all(axis=1))[0]
        id3 = np.where((data[:,0:3] == tri[2,:]).all(axis=1))[0]
        values[i] = (data[id1,3] + data[id2,3] + data[id3,3])/3

    normal = Normalize(vmin= np.min(data[:,3]), vmax = np.max(data[:,3]))
    colors = get_cmap(cmap)(normal(values))
    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)
    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=normal)


def plot_stacked_bar(data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, 
                     colors=None, grid=True, reverse=False):

    
    """Plots a stacked bar chart with the data and labels provided.

    Ref:https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib 
    
    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")