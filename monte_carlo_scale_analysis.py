import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from visualization_helper.helper import LinearNDInterpolatorExt, plot_stacked_bar
from scipy.integrate import odeint, solve_ivp
import pickle

# Turn of warning to avoid the later matplotlib versions complain about setting up ticks with a traditional way.
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

labelSize = 24
axisLabelSize = 12
plotEachScale = True
figList = []
scalePressStress = True # use the experiments that scale the pre-tension based on mass
folderName = "AnalysisResultPaper/"
filePrefix = "result_x10_"
scaleTextList = ["n1","n075","n05","n025","0","025","05","075","1"]
scalingFactorNames = ('$10^{-1}$','$10^{-3/4}$','$10^{-1/2}$','$10^{-1/4}$','$10^0$','$10^{1/4}$','$10^{1/2}$','$10^{3/4}$','$10^1$')


ratioLevel = [2**(i) for i in range(-1,3)]
sampleSize = len(scaleTextList)
ratioCount = np.zeros((sampleSize,len(ratioLevel)+1)) # 3 different ration categories, 0~1,1~2,2+

bin_range = (0, 15)
num_bins = 30

fig, axs = plt.subplots(3, 3, figsize=(20, 20),sharey=True)
boxplot_data = []

for i in range(sampleSize):
    row = i // 3
    col = i % 3
    ax = axs[row][col]
    scaleText = scaleTextList[i]
    file_O = open(folderName+filePrefix+scaleText+".pickle", 'rb')
    data_O = pickle.load(file_O)
    file_O.close()
    tensegrityData = data_O["tensegrity"]
    propGuardData = data_O["prop_guard"]
    n = tensegrityData.shape[0]
    ratioData = np.zeros((n,4))
    ratioData[:,0:3] = tensegrityData[:,0:3]
    ratioData[:,3] = (propGuardData[:,6]/tensegrityData[:,6]).squeeze()
    boxplot_data.append(ratioData[:,3])
    ax.hist(ratioData[:,3], bins=num_bins, range=bin_range)
    ax.set_title("Scaling Factor = "+scalingFactorNames[i], fontsize=axisLabelSize)
    ax.set_xlabel('Ratio of Max Stress: Prop-guard/Tensegrity', fontsize=axisLabelSize)
    ax.set_ylabel('Count', fontsize=axisLabelSize)
    ax.tick_params(axis='both', which='major', labelsize=axisLabelSize)
    ax.tick_params(axis='both', which='minor', labelsize=axisLabelSize)
    ax.xaxis.labelpad = -1
    
# Adjust the spacing between the subplots
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Rotate the x-axis tick labels
for ax in axs.flat:
    ax.tick_params(axis='x', labelrotation=45)


"""
Draw the box plot
"""
medianprops = dict(linestyle='--', linewidth=3, color='darksalmon')
meanprops = dict(marker='o', markeredgecolor='darksalmon', markerfacecolor='darksalmon')
whiskerprops = dict(linestyle='-', linewidth=2, color='gray')
flierprops=dict(marker= 'o', markersize= 5, markerfacecolor= 'darksalmon')
capprops =  dict(linestyle='-', linewidth=2, color='gray')

fig2, axs2 = plt.subplots(1, 1, figsize=(20, 6))
# Create the boxplot
bplot = axs2.boxplot(boxplot_data, patch_artist=True, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops,flierprops = flierprops, capprops=capprops)
for patch in bplot['boxes']:
    patch.set_facecolor('lightblue')
# Set the y-axis to log-log scale
axs2.set_yscale('log')

# Set custom y-axis ticks
y_ticks = [0, 10**(-1/2), 1, 10**(1/2), 10**1]
axs2.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
axs2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else '${{10^{{{:1.0f}/{:1.0f}}}}}$'.format(np.log10(x).as_integer_ratio()[0], np.log10(x).as_integer_ratio()[1]) if not np.log10(x).is_integer() else '${{10^{{{:1.0f}}}}}$'.format(np.log10(x))))

# Add labels and formatting
axs2.set_xticklabels(scalingFactorNames, fontsize=labelSize)
axs2.set_xlabel('Scaling Factor', fontsize=labelSize)
axs2.set_ylabel('Ratio of Max Stress', fontsize=labelSize)
axs2.tick_params(axis='both', labelsize=labelSize)
axs2.legend([bplot["medians"][0], bplot["fliers"][0]], ["Median", "Outliers"], fontsize=labelSize)
axs2.grid(True)
plt.savefig("Scale Analysis.pdf",  format="pdf", bbox_inches="tight")
plt.show()