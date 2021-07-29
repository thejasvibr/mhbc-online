# -*- coding: utf-8 -*-
"""
Making the multi-panel plot showing individual call measurements and 
mean estimates

"""
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import pandas as pd
import seaborn as sns

dB = lambda X: 20*np.log10(abs(X))

#%% 
# Load the individual call analysis results
results = pd.read_csv('../../combined_analysis/all_ind_summary.csv')
results = results.loc[:,'Component':'diff_upper_95pcHPD']
# Load the Mean MAP and 95% compatibility interval 
mean_estimates = pd.read_csv('../../combined_analysis/single-multi_estimatedmeans.csv')

#%%
ind_call = pd.read_csv('../../combined_analysis/indcall_final.csv')

ind_call['groupsize'] = 'NA'
for i,each in ind_call.iterrows():
    if each['num_bats']>1:
        ind_call.loc[i,'groupsize'] = 'z_multi'
    else:
        ind_call.loc[i,'groupsize'] = 'a_single'


multibat_indcall = ind_call.copy()
for col in ['ifm_rms','tfm_rms','cf_rms']:
    multibat_indcall[col+'db'] = dB(multibat_indcall[col])

multibat_indcall['ifm_bw'] = ind_call['cf_peak_frequency'] - ind_call['ifm_terminal_frequency']
multibat_indcall['tfm_bw'] = ind_call['cf_peak_frequency'] - ind_call['tfm_terminal_frequency']

groupsize_dict = {}
for each in [2,3,4]:
    groupsize_dict[each] = 'multi'
groupsize_dict[1] = 'single'

multibat_indcall['groupsize'] = multibat_indcall['num_bats'].apply(lambda X: groupsize_dict[X])

def remove_top_right_spines():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
def remove_three_spines():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

def plot_map_compint(df_row, xpos=[0.55, 1.55]):
    '''
    '''
    err1 =  np.array([df_row['single-95hpd-lo'], df_row['single-95hpd-hi']]).reshape(2,1) - df_row['single-map']
    err2 =  np.array([df_row['multi-95hpd-lo'], df_row['multi-95hpd-hi']]).reshape(2,1) - df_row['multi-map']
    
    dotsize = 2
    linethickness = 0.9
    # the dot
    
    plt.plot(xpos[0],df_row['single-map'],'o',color='blue', markersize=dotsize)
    plt.plot(xpos[1],df_row['multi-map'],'^',color='orange', markersize=dotsize)
    # the vertical line
    plt.errorbar(xpos[0],df_row['single-map'], yerr=err1, linewidth=linethickness)
    plt.errorbar(xpos[1],df_row['multi-map'], yerr=err2, linewidth=linethickness)

#%% 

one_column_size = (3.5,5)
BIGGGSIZE = (6,4)
fig2e = plt.figure(figsize=one_column_size)
# DONT have constrained_layout=True --- THIS MAKES A HUUUUGE DIFFERENCE!!
spec2e = gridspec.GridSpec(ncols=48, nrows=54, figure=fig2e)
spec2e.update(wspace=0.05,hspace=0.02) # set the spacing between axes. 

row1 = list(range(10))
row2 = list(range(10,20))
row3 = list(range(20,30))
row4 = list(range(30,40))
row5 = list(range(40,50))

col1 = list(range(1,11))
col2 = list(range(13,23))
col3 = list(range(25,35))
col4 = list(range(37,48))



f2e_ax1 = fig2e.add_subplot(spec2e[row1[0]:row1[-1], col1[0]:col1[-1]])
f2e_ax12 = fig2e.add_subplot(spec2e[row1[0]:row1[-1], col2[0]:col2[-1]])
f2e_ax2 = fig2e.add_subplot(spec2e[row1[0]:row1[-1], col3[0]:col3[-1]])
f2e_ax3 = fig2e.add_subplot(spec2e[row1[0]:row1[-1], col4[0]:col4[-1]])

f2e_ax5 = fig2e.add_subplot(spec2e[row2[0]:row2[-1], col1[0]:col1[-1]])
f2e_ax6 = fig2e.add_subplot(spec2e[row2[0]:row2[-1], col3[0]:col3[-1]])
f2e_ax7 = fig2e.add_subplot(spec2e[row2[0]:row2[-1], col4[0]:col4[-1]])

f2e_ax9 = fig2e.add_subplot(spec2e[row3[0]:row3[-1], col1[0]:col1[-1]])
f2e_ax10 = fig2e.add_subplot(spec2e[row3[0]:row3[-1], col3[0]:col3[-1]])
f2e_ax11 = fig2e.add_subplot(spec2e[row3[0]:row3[-1], col4[0]:col4[-1]])

f2e_ax14 = fig2e.add_subplot(spec2e[row4[0]:row4[-1], col3[0]:col3[-1]])
f2e_ax15 = fig2e.add_subplot(spec2e[row4[0]:row4[-1], col4[0]:col4[-1]])


f2e_ax18 = fig2e.add_subplot(spec2e[row5[0]:row5[-1],col3[0]:col3[-1]])
f2e_ax19 = fig2e.add_subplot(spec2e[row5[0]:row5[-1],col4[0]:col4[-1]])

#f2e_ax20 = fig2e.add_subplot(spec2e[4,0])

point_size = 1.1

ylabx, ylaby = -.75, -.1
ylab_fontsize=6
yticks_fontsize=6

newcallpart_labelx, newcallpart_labely = 0.45, 1.1

def make_single_multi_labels():
    plt.text(-0.25, -0.25, 'single', fontsize=yticks_fontsize, transform=plt.gca().transAxes)
    plt.text(0.55, -0.25, 'multi', fontsize=yticks_fontsize, transform=plt.gca().transAxes)

def make_subplotlabel(axesname, letter,subplotx=0.9, subploty=0.8):
    plt.text(subplotx, subploty, letter, transform=axesname.transAxes,
                             fontsize=6)
#plt.show()
 
plt.sca(f2e_ax1) 
plt.text(newcallpart_labelx, newcallpart_labely, 'CF', transform=plt.gca().transAxes, fontsize=6,weight='bold')
plt.sca(f2e_ax2)
f2e_ax2.text(newcallpart_labelx, newcallpart_labely, 'tFM', transform=plt.gca().transAxes, fontsize=6,weight='bold')
plt.sca(f2e_ax3)
f2e_ax3.text(newcallpart_labelx, newcallpart_labely, 'iFM', transform=plt.gca().transAxes, fontsize=6,weight='bold')

#%% Temporal
plt.sca(f2e_ax1)
remove_three_spines()
sns.boxplot(y='cf_duration',x='groupsize', data=multibat_indcall,
            order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0 )
sns.swarmplot(y='cf_duration',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size,alpha=0.4)
plt.ylim(0,60)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.text(ylabx, ylaby, 'Duration (ms)', transform=f2e_ax1.transAxes,fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
f2e_ax1.tick_params(axis='y', which='major', pad=0.025);plt.yticks([0,25,50],['0','','50'],fontsize=yticks_fontsize)
make_subplotlabel(plt.gca(),'A1');plt.xlim(-0.5,1.7)

plot_map_compint(mean_estimates.loc[0,:])

plt.sca(f2e_ax2)
remove_three_spines()
sns.swarmplot(y='tfm_duration',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='tfm_duration',x='groupsize', data=multibat_indcall,
            order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
# plt.text(ylabx, ylaby+0.2, 'Duration\n(ms)', transform=plt.gca().transAxes, fontsize=11, rotation='vertical',multialignment='center')
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.ylim(0,6)
plt.text(ylabx, ylaby, '', transform=f2e_ax2.transAxes,
                              fontsize=11, rotation='vertical')
f2e_ax2.tick_params(axis='y', which='major', pad=0.025);
plt.yticks([0,2.5,5],['0','','5'],fontsize=yticks_fontsize)
make_subplotlabel(plt.gca(),'B');plt.xlim(-0.5,1.7)
plot_map_compint(mean_estimates.loc[1,:])

plt.sca(f2e_ax3)
remove_three_spines()
sns.swarmplot(y='ifm_duration',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='ifm_duration',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
plt.yticks([0,2.5,5],['0','','5'],fontsize=yticks_fontsize)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
#plt.text(ylabx, ylaby+0.2, 'Duration\n(ms)', transform=plt.gca().transAxes, fontsize=11, rotation='vertical',multialignment='center')
plt.ylim(0,6)
plt.text(ylabx, ylaby, '', transform=f2e_ax3.transAxes,
                              fontsize=11, rotation='vertical')
f2e_ax3.axes.yaxis.set_ticklabels([])
f2e_ax3.tick_params(axis='y', which='major', pad=0.025);plt.yticks([0,2.5,5],fontsize=yticks_fontsize)
make_subplotlabel(plt.gca(),'C');plt.xlim(-0.5,1.7)
plot_map_compint(mean_estimates.loc[2,:])


#%% Spectral
plt.sca(f2e_ax5)
remove_three_spines()
sns.swarmplot(y='cf_peak_frequency',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='cf_peak_frequency',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.text(ylabx, ylaby, ' Peak frequency\n(kHz)', transform=f2e_ax5.transAxes,
                              fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
plt.yticks([100,112],fontsize=yticks_fontsize);plt.ylim(100,112.5)
f2e_ax5.tick_params(axis='y', which='major', pad=0.025)
make_subplotlabel(plt.gca(),'D'); plt.xlim(-0.5,1.7)
plot_map_compint(mean_estimates.loc[3,:])


plt.sca(f2e_ax6)
remove_three_spines()
sns.swarmplot(y='tfm_terminal_frequency',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='tfm_terminal_frequency',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.ylim(80,111); plt.yticks([80,100,120],['80','','120'], fontsize=yticks_fontsize)
f2e_ax6.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
make_subplotlabel(plt.gca(),'E')
plt.text(ylabx-0.05, ylaby, 'Lower frequency\n(kHz)', transform=plt.gca().transAxes,
         fontsize=ylab_fontsize, rotation='vertical', multialignment='center')
plot_map_compint(mean_estimates.loc[4,:])



plt.sca(f2e_ax7)
remove_three_spines()
sns.swarmplot(y='ifm_terminal_frequency',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='ifm_terminal_frequency',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
# plt.text(ylabx, ylaby, 'Lower frequency\n(kHz)', transform=plt.gca().transAxes, fontsize=11, rotation='vertical', multialignment='center')
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.text(ylabx, ylaby, '', transform=f2e_ax7.transAxes,
                              fontsize=11, rotation='vertical')
plt.ylim(80,111); plt.yticks([80,100,120],['80','','120'], fontsize=yticks_fontsize)
f2e_ax7.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
f2e_ax7.axes.yaxis.set_ticklabels([])
make_subplotlabel(plt.gca(),'F')
plot_map_compint(mean_estimates.loc[5,:])

#%% Received level

plt.sca(f2e_ax9)
remove_top_right_spines()
sns.swarmplot(y='cf_rmsdb',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='cf_rmsdb',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
plt.xlabel('');plt.ylabel('')
plt.text(ylabx, ylaby+0.05, 'Received level\n (dB rms)', transform=plt.gca().transAxes,
                              fontsize=ylab_fontsize, rotation='vertical', multialignment='center')
plt.yticks([-42,-24,-6],['-42','','-6'], fontsize=yticks_fontsize);plt.ylim(-42,-6)
f2e_ax9.axes.xaxis.set_ticklabels([]);

make_single_multi_labels()

#plt.xticks([-0.35, 1.1], ['single','multi'],fontsize=yticks_fontsize)
f2e_ax9.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
make_subplotlabel(plt.gca(),'G')
plot_map_compint(mean_estimates.loc[6,:])


plt.sca(f2e_ax10)
remove_three_spines()
sns.swarmplot(y='tfm_rmsdb',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='tfm_rmsdb',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
# plt.text(ylabx, ylaby, 'Received level\n(dB rms)', transform=plt.gca().transAxes, fontsize=11, rotation='vertical', multialignment='center')
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.ylim(-54,-10);plt.yticks([-50, -30, -10],['-50','','-10'], fontsize=yticks_fontsize)
plt.text(ylabx, ylaby, '', transform=f2e_ax10.transAxes, fontsize=11, rotation='vertical')
f2e_ax10.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
make_subplotlabel(plt.gca(),'H')
plot_map_compint(mean_estimates.loc[7,:])


plt.sca(f2e_ax11)
remove_three_spines()
sns.swarmplot(y='ifm_rmsdb',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='ifm_rmsdb',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
#  plt.text(ylabx, ylaby, 'Received level\n(dB rms)', transform=plt.gca().transAxes,  fontsize=11, rotation='vertical', multialignment='center')
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.ylim(-54,-10);plt.ylabel('');plt.yticks([-50, -30, -10], fontsize=yticks_fontsize)
plt.text(ylabx, ylaby, '', transform=f2e_ax11.transAxes,
                              fontsize=11, rotation='vertical')
f2e_ax11.axes.yaxis.set_ticklabels([])
f2e_ax11.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
make_subplotlabel(plt.gca(),'I')
plot_map_compint(mean_estimates.loc[8,:])

#%% derived parameters
plt.sca(f2e_ax14)
remove_three_spines()
sns.swarmplot(y='tfm-cf_dbratio',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='tfm-cf_dbratio',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)

plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.ylim(-20,6);plt.yticks([-18,-6,6], ['-18','','6'],fontsize=yticks_fontsize)
plt.text(ylabx, ylaby, 'tFM-CF $\Delta$ level (dB)', transform=f2e_ax14.transAxes,
                              fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
f2e_ax14.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
make_subplotlabel(plt.gca(),'J')
plot_map_compint(mean_estimates.loc[9,:])


plt.sca(f2e_ax15)
remove_three_spines()
sns.swarmplot(y='ifm-cf_dbratio',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='ifm-cf_dbratio',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
plt.xlabel(''); plt.xticks([]);plt.ylabel('')
plt.ylim(-20,6);plt.yticks([-18,-6,6], ['-18','','6'], fontsize=yticks_fontsize)
plt.text(ylabx, ylaby, 'iFM-CF $\Delta$ level (dB)', transform=f2e_ax15.transAxes,
                              fontsize=ylab_fontsize, rotation='vertical',multialignment='center')
f2e_ax15.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
plt.gca().axes.yaxis.set_ticklabels([])
make_subplotlabel(plt.gca(),'K')
plot_map_compint(mean_estimates.loc[10,:])


callpart_labelx, callpart_labely = 0.4, -0.6

plt.sca(f2e_ax18)
remove_top_right_spines()
sns.swarmplot(y='tfm_bw',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='tfm_bw',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)


plt.ylim(0,24);plt.ylabel('');plt.yticks([0,12,24],['0','','24'],fontsize=yticks_fontsize)

plt.text(ylabx, ylaby, 'Bandwidth (kHz)', transform=plt.gca().transAxes,
                              fontsize=ylab_fontsize, rotation='vertical', multialignment='center')
f2e_ax18.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
make_subplotlabel(plt.gca(),'L')
plot_map_compint(mean_estimates.loc[11,:])
f2e_ax18.axes.xaxis.set_ticklabels([]);plt.xlabel('');

make_single_multi_labels()

plt.sca(f2e_ax19)
remove_top_right_spines()
sns.swarmplot(y='ifm_bw',x='groupsize', data=multibat_indcall, order=['single','multi'],size=point_size, alpha=0.4)
sns.boxplot(y='ifm_bw',x='groupsize', data=multibat_indcall,
             order=['single','multi'], color='white', showfliers=False, width=0.5,
            linewidth=0.8, whis=0)
 # plt.text(ylabx, ylaby+0.2, 'Bandwidth\n(kHz)', transform=plt.gca().transAxes, fontsize=11, rotation='vertical', multialignment='center')
plt.ylabel('');plt.ylim(0,24);plt.ylabel('');plt.yticks([0,12,24],fontsize=yticks_fontsize)
plt.xlabel('');
plt.text(ylabx, ylaby, '', transform=f2e_ax19.transAxes,
                              fontsize=11, rotation='vertical')
f2e_ax19.tick_params(axis='y', which='major', pad=0.025);plt.xlim(-0.5,1.7)
plt.gca().axes.yaxis.set_ticklabels([])

make_subplotlabel(plt.gca(),'M')
plot_map_compint(mean_estimates.loc[12,:])
make_single_multi_labels()
f2e_ax19.axes.xaxis.set_ticklabels([]);plt.xlabel('');


 #plt.sca(f2e_ax20)
 #plt.axis('off')
plt.savefig('measurements_and_derivedparams_multipanel.png')

