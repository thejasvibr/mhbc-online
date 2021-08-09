# -*- coding: utf-8 -*-
"""
Module that plots the raw windowed analysis measurements along with 
the estimated mean MAP and compatibility interval. 


@author: tbeleyur
"""


import os
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import pandas as pd
import seaborn as sns

dB = lambda X: 20*np.log10(abs(X))

#%% load the measurements
analysis_folder = '../../combined_analysis'
reclevel = pd.read_csv(os.path.join(analysis_folder,'obsvirt_reclevel.csv'))
lowerfreq = pd.read_csv(os.path.join(analysis_folder,'obsvirt_lowerfreq.csv'))
domfrange = pd.read_csv(os.path.join(analysis_folder, 'obsvirt_domfrange.csv'))
#%% load the statistical mean MAP and Co. I estiamtes
mean_mapcoi = pd.read_csv(os.path.join(analysis_folder,
                                       'windowed_mean-map-coi-table.csv'))

#%% Convenience functions 

def remove_top_right_spines():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
def remove_three_spines():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

def plot_map_compint(df_row, xpos=[0.46, 1.46, 2.46]):
    '''
    '''
    err1 =  np.array([df_row['single-95hpd-lower'], df_row['single-95hpd-upper']]).reshape(2,1)
    err1 += - df_row['single-map']
    err2 =  np.array([df_row['multi-95hpd-lower'], df_row['multi-95hpd-upper']]).reshape(2,1)
    err2 += -df_row['multi-map']
    err3 = np.array([df_row['virtmulti-95hpd-lower'], df_row['virtmulti-95hpd-upper']]).reshape(2,1)
    err3 += -df_row['virtmulti-map']
    
    dotsize = 2
    linethickness = 0.9
    # the dot
    
    plt.plot(xpos[0],df_row['single-map'],'o',color='blue', markersize=dotsize)
    plt.plot(xpos[1],df_row['multi-map'],'o',color='orange', markersize=dotsize)
    plt.plot(xpos[2],df_row['virtmulti-map'],'o',color='green', markersize=dotsize)
    
    # the vertical line
    plt.errorbar(xpos[0],df_row['single-map'], yerr=np.abs(err1), linewidth=linethickness)
    plt.errorbar(xpos[1],df_row['multi-map'], yerr=np.abs(err2), linewidth=linethickness)
    plt.errorbar(xpos[2],df_row['virtmulti-map'], yerr=np.abs(err3), linewidth=linethickness)


def make_single_multi_labels():
    plt.text(0.15, -0.25, 'single', fontsize=yticks_fontsize, transform=plt.gca().transAxes)
    plt.text(0.45, -0.25, 'multi', fontsize=yticks_fontsize, transform=plt.gca().transAxes)
    plt.text(0.75, -0.35, 'virtual\nmulti', fontsize=yticks_fontsize, transform=plt.gca().transAxes)

def make_subplotlabel(axesname, letter,subplotx=0.85, subploty=0.95):
    plt.text(subplotx, subploty, letter, transform=axesname.transAxes,
                             fontsize=6, multialignment='center')

def common_xlim():
    plt.xlim(-0.7, 2.7)

#%% Plot raw data in a 3 x 1 plot with data points, boxplot and 
# Mean estimate bars. 
    

one_column_size = (3.5,3.5)
fig2e = plt.figure(figsize=one_column_size)
# DONT have constrained_layout=True --- THIS MAKES A HUUUUGE DIFFERENCE!!
spec2e = gridspec.GridSpec(ncols=17, nrows=28, figure=fig2e)    

row1 = list(range(9))
row2 = list(range(10,19))
row3 = list(range(20,27))

col1 = list(range(1,17))
col2 = list(range(1,17))
col3 = list(range(1,17))

ax00 = fig2e.add_subplot(spec2e[row1[0]:row1[-1], col1[0]:col1[-1]])
ax10 = fig2e.add_subplot(spec2e[row2[0]:row2[-1], col2[0]:col2[-1]])
ax20 = fig2e.add_subplot(spec2e[row3[0]:row3[-1], col3[0]:col3[-1]])

#%% Plot parameters

point_size = 1.1

ylabx, ylaby = -.15, -0.28
ylab_fontsize=6
yticks_fontsize=6

#%% Dominant frquency range
plt.sca(ax00)

remove_three_spines()

sns.boxplot(y='domfrange',x='group_status', data=domfrange,
            order=[0,1,2], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0 )
sns.swarmplot(y='domfrange',x='group_status', data=domfrange,
              order=[0,1,2],size=point_size,alpha=0.4)
plt.ylim(-0.5,8.5);
plt.yticks([0,4,8],[0,'',8],fontsize=yticks_fontsize)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.text(ylabx, ylaby+0.25, 'Dominant frequency\nrange (kHz)', transform=ax00.transAxes,fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
plot_map_compint(mean_mapcoi.loc[0,:])
ax00.tick_params(axis='y', which='major', pad=0.025);
make_subplotlabel(plt.gca(),'A')
common_xlim()

#%% lower frequency 
plt.sca(ax10)
remove_three_spines()

sns.boxplot(y='lowerfreqkhz',x='group_status', data=lowerfreq,
            order=[0,1,2], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0 )
sns.swarmplot(y='lowerfreqkhz',x='group_status', data=lowerfreq,
              order=[0,1,2],size=point_size,alpha=0.4)
plt.ylim(70.5,100.5);
plt.yticks([70,85,100],[70,'',100],fontsize=yticks_fontsize)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.text(ylabx, ylaby+0.25, 'FM lower frequency\n(kHz)', transform=ax10.transAxes,fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
plot_map_compint(mean_mapcoi.loc[1,:])
ax00.tick_params(axis='y', which='major', pad=0.025);
make_subplotlabel(plt.gca(),'B')
common_xlim()

#%% received level
plt.sca(ax20)
remove_top_right_spines()

sns.boxplot(y='dbrms',x='group_status', data=reclevel,
            order=[0,1,2], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0 )
sns.swarmplot(y='dbrms',x='group_status', data=reclevel,
              order=[0,1,2],size=point_size,alpha=0.4)
plt.ylim(-38.5,-6.0);
plt.yticks([-38,-22,-6],[-38,'',-6],fontsize=yticks_fontsize)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.text(ylabx, ylaby+0.25, 'Received level \n(dB rms)', transform=ax20.transAxes,fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
plot_map_compint(mean_mapcoi.loc[2,:])
ax00.tick_params(axis='y', which='major', pad=0.025);
make_subplotlabel(plt.gca(),'C')
common_xlim()
make_single_multi_labels()
plt.savefig('window_analysis_results.png', bbox_inches='tight', dpi=600)