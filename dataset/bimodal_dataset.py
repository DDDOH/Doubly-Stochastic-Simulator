
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--reproduce', type=bool, default=True)
args = parser.parse_args()
REPRODUCE = args.reproduce
# if REPRODUCE: simply load the saved dataset in reproduce folder and make some visualizations.
# else: generate a new dataset and save it in dataset folder.

# files in reproduce are not changed by any codes.
P = 16  # works only when TRAIN == True

########## For Figure 7, bimodal synthetic dataset ##########
if not REPRODUCE: # generate a new dataset and save it in dataset folder.
    
    from scipy.stats import (norm, multivariate_normal, gamma, pareto, uniform)
    import os
    TRAIN = True
    
    SEED_DIM = 4  # works only when TRAIN == True
    NOISE = 'normal'  # uniform or normal
    ONLY_MLP = True
    lr_final = 1e-4
    lr_initial = 0.001
    N_TRAIN = 700

    DEBUG = False
    if DEBUG:
        ITERS = 300
        SAVE_FREQ = 100
    else:
        ITERS = 30000
        SAVE_FREQ = 1000
        
    # set marginal mean, marginal variance for intensity.
    # set corr_mat for the underlying multi-normal distribution
    CC_train = np.load('dataset/callcenter/callcenter.npy')

    data = CC_train
    p = np.shape(data)[1]
    corr_mat = np.corrcoef(data, rowvar=False)

    marginal_mean = np.mean(data, axis=0)
    marginal_var = np.var(data, axis=0)

    assert P <= p

    modal1_mean = marginal_mean[:P] * 30
    modal2_mean = marginal_mean[:P] * 30 + marginal_mean[:P] * 15 * np.random.uniform(-1, 1.5, P)

    # a randomly generated psd matrix
    A = np.random.rand(P, P)
    B = np.dot(A, A.transpose())

    modal1_cov = np.cov(data, rowvar=False)
    modal2_cov = B * marginal_var * np.random.uniform(0.8, 1.2, size=(P))

    prob_modal_1 = 0.7

    u = np.random.rand(N_TRAIN)

    intensity_modal_1 = multivariate_normal.rvs(mean=modal1_mean, cov=modal1_cov, size=N_TRAIN)
    intensity_modal_2 = multivariate_normal.rvs(mean=modal2_mean, cov=modal2_cov, size=N_TRAIN)

    intensity = np.empty_like(intensity_modal_1)
    intensity[u < prob_modal_1] = intensity_modal_1[u < prob_modal_1]
    intensity[u >= prob_modal_1] = intensity_modal_2[u >= prob_modal_1]

    train = np.random.poisson(intensity)


    data_dir = 'dataset/margin_bimodal__P_16'
    # create a folder to store the data
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    np.save(os.path.join(data_dir, 'train_bimodal.npy'), train)
    np.save(os.path.join(data_dir, 'intensity_bimodal.npy'), intensity)

else:
    train = np.load('reproduce/reproduce_margin_bimodal__P_16/train_bimodal.npy')


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


# SMALL_SIZE = 14

AXES_TITLE_SIZE = 5 # 没啥用
AXES_LABEL_SIZE = 16 # xy轴名称的大小
TICK_SIZE = 12 # x y轴数字的大小
LEGEND_SIZE = 12
# BIGGER_SIZE = 16

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=AXES_TITLE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXES_LABEL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, ax = plt.subplots()
plt.plot(np.mean(train,axis=0))
plt.scatter(np.tile(np.arange(P), np.shape(train)[0]).reshape(
        P, np.shape(train)[0]), train, alpha=0.02, c='C1', s=40, edgecolors='none',label=r'$(j,X_{i,j})$')
plt.xlabel('Time Interval')
plt.ylabel('Arrival Count')
plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
legend_elements = [Line2D([0], [0], color='C0', lw=1, label='Marginal Mean'),
                Line2D([0], [0], marker='o', lw=0, color='w', label=r'$(j,X_{i,j})$',
                        markerfacecolor='C1', markersize=7, alpha=0.6)]

# Create the figure
# plt.subplots()
ax.legend(handles=legend_elements)
# plt.legend()
plt.tight_layout()
fig_dir = 'reproduce/reproduce_margin_bimodal__P_16/' if REPRODUCE else 'dataset/margin_bimodal__P_16/'
plt.savefig(os.path.join(fig_dir, 'bimodal_train_mean_scatter.pdf'))
plt.close('all')


plt.figure()
plt.plot(np.var(train,axis=0))
plt.xlabel('Time Interval')
plt.ylabel('Marginal Variance of Arrival Count')
plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'bimodal_train_var.pdf'))
plt.close('all')