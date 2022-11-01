import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.colors as colors
from visualization_helper.helper import LinearNDInterpolatorExt, plot_stacked_bar, Octant_Mapping
from scipy.integrate import odeint, solve_ivp
import pickle
import seaborn as sns
sns.set_theme()
# Turn of deprecation warning due to seaborn-matplotlib deprecation conflict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

usePreparedData = True
if usePreparedData:
    #Prepare data intepretator 
    folderName = "AnalysisResultPaper/"
    scaleText = "0"
    filePrefix = "result_x10_"
    file_O = open(folderName+filePrefix+scaleText+".pickle", 'rb')
    data_O = pickle.load(file_O)
    file_O.close()
else: 
    folderName = "AnalysisResult/"
    scale = 1
    scaleText = "scale_"+str(scale)
    filePrefix = "result_"
    file_O = open(folderName+filePrefix+scaleText+".pickle", 'rb')
    data_O = pickle.load(file_O)
    file_O.close()

#Graph format setups
format = '%.2f'
colorBarTickFormat = ticker.FormatStrFormatter(format)
plotContour = False
labelSize = 50
tickSize = 13

tensegrityData = data_O["tensegrity"]
propGuardData = data_O["prop_guard"]

n = tensegrityData.shape[0]
ratioData = np.zeros((n,4))
pointsXYZ = tensegrityData[:,0:3] #scattered sample points in x-y-z cartesian coordinate

pointsPolar = np.zeros((tensegrityData.shape[0],2)) #scattered sample points in polar coordinate
pointsPolar[:,0] = np.arctan2(pointsXYZ[:,1],pointsXYZ[:,0])
pointsPolar[:,1] = np.arccos(pointsXYZ[:,2])
values = (propGuardData[:,6]/tensegrityData[:,6]).squeeze()

ratioData[:,0:3] = pointsXYZ
ratioData[:,3] = values
linInterExt= LinearNDInterpolatorExt(pointsPolar, values) #A linear interpolator/extrapolator

# Find the distribution of the sampled data
ratioLevel = [1,2,4,6]

# [2**(i) for i in range(0,4)] #[0~0.5],[0.5~1],[1~2],[2~4],[4~]
ratioPercentage = np.zeros(len(ratioLevel)+1) 
totalNum = ratioData.shape[0]
remainedData = ratioData
for i in range(len(ratioLevel)):
    if remainedData.shape[0] <=0:
        continue
    dataInRange = remainedData[np.where(remainedData[:,3]<=ratioLevel[i])]
    ratioPercentage[i] = dataInRange.shape[0]/totalNum
    remainedData = remainedData[np.where(remainedData[:,3]>ratioLevel[i])]
ratioPercentage[-1] = remainedData.shape[0]/totalNum

print(str(ratioPercentage[0]*100)+"% of all samples has a max stress ratio in [0," +str(ratioLevel[0])+"]")
for i in range(len(ratioLevel)-1):
    print(str(ratioPercentage[i+1]*100)+"% of all samples has a max stress ratio in ["+str(ratioLevel[i])+ ","+str(ratioLevel[i+1])+"]")
print(str(ratioPercentage[-1]*100)+"% of all samples has a max stress ratio in ["+str(ratioLevel[-1])+ ", inf]")

# Interpoloate/extrapolate scattered datapoints to a polar grid
n_theta = 250 # number of values for theta
n_phi = 250  # number of values for phi
theta, phi = np.mgrid[np.pi/2:np.pi:n_theta*1j, 0.0:np.pi/2:n_phi*1j]

# Prepare grid data
x_g = np.sin(phi) * np.cos(theta)
y_g = np.sin(phi) * np.sin(theta)
z_g = np.cos(phi)
v_g = np.zeros((n_theta, n_phi)) # value grid
for i in range(n_theta):
    for j in range(n_phi):
        polarCoord = np.array([theta[i,j],phi[i,j]])
        v_g[i,j] = linInterExt(polarCoord)

print("meanPropGuardStress=", np.mean(propGuardData[:,6]))
print("meanTensegrity=", np.mean(tensegrityData[:,6]))
print("maxPropGuardStress=", np.max(propGuardData[:,6]))
print("maxTensegrity=", np.max(tensegrityData[:,6]))

"""
Fig top: max stress scatter plot for both tensegrity and prop guard
"""
minStress = min([np.min(propGuardData[:,6]), np.min(tensegrityData[:,6])])
maxStress = max([np.max(propGuardData[:,6]), np.max(tensegrityData[:,6])])
normStress = plt.Normalize(minStress, maxStress)

fig0 = plt.figure(figsize=(15,15), dpi=300)
ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
ax0.grid(False)
cs0 = ax0.scatter(tensegrityData[:,0], tensegrityData[:,1], tensegrityData[:,2], c = tensegrityData[:,6], cmap = "cool", lw=0, s=60, norm = normStress)

ax0.set_xlim3d([-1.0, 0.2])
ax0.set_xlabel("${e_x}^B$",fontsize=labelSize,labelpad=0)
ax0.set_ylim3d([-0.2, 1.0])
ax0.set_ylabel("${e_y}^B$",fontsize=labelSize,labelpad=0)
ax0.set_zlim3d([-0.2, 1.0])
ax0.set_zlabel("${e_z}^B$",fontsize=labelSize,labelpad=0)

ax0.set_title('Tensegrity Maximum Stress [Pa]')
ax0.view_init(25, 135)
ax0.set_facecolor('white')

# Remove ticks
ax0.xaxis.set_tick_params(color='white')
ax0.yaxis.set_tick_params(color='white')
ax0.zaxis.set_tick_params(color='white')
ax0.axes.xaxis.set_ticklabels([])
ax0.axes.yaxis.set_ticklabels([])
ax0.axes.zaxis.set_ticklabels([])


cbar0 = fig0.colorbar(cs0, ax=ax0,orientation='vertical',shrink=0.67, aspect=16.7)
cbar0.ax.tick_params(labelsize=tickSize)
cbar0.ax.yaxis.get_offset_text().set(size=tickSize)

plt.savefig("TensegrityMaxStress.svg", format = 'svg', dpi=300)
plt.savefig("TensegrityMaxStress.png", format = 'png', dpi=300)

fig1 = plt.figure(figsize=(15,15), dpi=300)
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')

cs1 = ax1.scatter(propGuardData[:,0], propGuardData[:,1], propGuardData[:,2], c=propGuardData[:,6], cmap = "cool", lw=0, s=75, norm = normStress)

ax1.set_xlim3d([-1.0, 0.2])
ax1.set_xlabel("${e_x}^B$",fontsize=labelSize,labelpad=0)
ax1.set_ylim3d([-0.2, 1.0])
ax1.set_ylabel("${e_y}^B$",fontsize=labelSize,labelpad=0)
ax1.set_zlim3d([-0.2, 1.0])
ax1.set_zlabel("${e_z}^B$",fontsize=labelSize,labelpad=0)
ax1.set_title('Prop-guard Maximum Stress [Pa]')
ax1.view_init(25, 135)

# Remove ticks
ax1.xaxis.set_tick_params(color='white')
ax1.yaxis.set_tick_params(color='white')
ax1.zaxis.set_tick_params(color='white')
ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])
ax1.axes.zaxis.set_ticklabels([])

ax1.set_facecolor('white')
ax1.grid(False)
cbar1 = fig1.colorbar(cs1, ax=ax1,orientation='vertical',shrink=0.67, aspect=16.7)
cbar1.ax.tick_params(labelsize=tickSize)
cbar1.ax.yaxis.get_offset_text().set(size=tickSize)

plt.savefig("PropGuardMaxStress.svg", format = 'svg', dpi=300)
plt.savefig("PropGuardMaxStress.png", format = 'png', dpi=300)

"""
Fig bottom: stress ratio plot
"""
colorMap = plt.cm.plasma_r  # define the colormap
# extract all colors the map
cmaplist = [colorMap(i) for i in range(colorMap.N)]
# create the new map
customCMap = colors.LinearSegmentedColormap.from_list('Custom', cmaplist, colorMap.N)
# define the bins and normalize

minRatio = min(ratioData[:,3])
maxRatio = max(ratioData[:,3])

# guard against cases when sample is small and all ratio falls in the 1~6 region.
if minRatio < 1 and maxRatio > 6:
    bounds = np.array([minRatio,1,2,3,4,5,6,maxRatio])
else: 
    bounds = np.array([1,2,3,4,5,6])
boundNorm = colors.BoundaryNorm(bounds, customCMap.N)

#Use this for continuouse mapping 
#contNorm = colors.Normalize(vmin= np.min(ratioData[:,3]), vmax = np.max(ratioData[:,3]))

fig2 = plt.figure(figsize=(15,15), dpi=300)
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.grid(False)
scamap = ScalarMappable(cmap=customCMap,norm=boundNorm)
fcolors = scamap.to_rgba(v_g)
heatMap = ax2.plot_surface(x_g,y_g,z_g, rstride=1, cstride=1, facecolors=fcolors, linewidth=1,shade=False) 

# Add contour mapped from 2d to 3d to the surface
if plotContour:
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(1, 1, 1)
    levels = bounds
    ct = ax_temp.contour(theta, phi, v_g, levels, colors=('k',),linestyles=('-',),linewidths=(2,))
    plt.clabel(ct, colors = 'k', fmt = '%2.1f', fontsize=labelSize)
    oct = Octant_Mapping()
    oct.octantContour(ax2,ct)
    fig_temp.clf()
    plt.close()

cs2 = ax2.scatter(ratioData[:,0], ratioData[:,1], ratioData[:,2], c = ratioData[:,3], cmap = customCMap, edgecolors='black', linewidths=0.2, s=25, alpha=0.75, norm = boundNorm)
cbar2 = fig2.colorbar(scamap, spacing='proportional', shrink=0.67, aspect=16.7)
cbar2.ax.tick_params(labelsize=tickSize)
cbar2.ax.yaxis.get_offset_text().set(size=tickSize)
cbar2.ax.yaxis.set_major_formatter(colorBarTickFormat)

ax2.set_xlim3d([-1.0, 0.2])
ax2.set_xlabel("${e_x}^B$",fontsize=labelSize,labelpad=0)
ax2.set_ylim3d([-0.2, 1.0])
ax2.set_ylabel("${e_y}^B$",fontsize=labelSize,labelpad=0)
ax2.set_zlim3d([-0.2, 1.0])
ax2.set_zlabel("${e_z}^B$",fontsize=labelSize,labelpad=0)
ax2.view_init(25, 135)
ax2.set_facecolor('white')
ax2.set_title('Scale:'+scaleText+'Ratio of Maximum Stress: Prop-guard/Tensegrity [Pa/Pa]')

# Remove ticks
ax2.xaxis.set_tick_params(color='white')
ax2.yaxis.set_tick_params(color='white')
ax2.zaxis.set_tick_params(color='white')
ax2.axes.xaxis.set_ticklabels([])
ax2.axes.yaxis.set_ticklabels([])
ax2.axes.zaxis.set_ticklabels([])

plt.savefig("MaxStressRatio.svg", format = 'svg', dpi=300)
plt.savefig("MaxStressRatio.png", format = 'png', dpi=300)