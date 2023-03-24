from utils import get_CI, get_corr
from colored import fg, attr
import numpy as np
import progressbar
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_label', type=str, default='pgnorta')
# pgnorta, pgnorta_gan

args = parser.parse_args()
dataset = args.exp_label

# SMALL_SIZE = 14

AXES_TITLE_SIZE = 5 # 没啥用
AXES_LABEL_SIZE = 16 # xy轴名称的大小
TICK_SIZE = 12 # x y轴数字的大小
LEGEND_SIZE = 12
TITLE_SIZE = 16
# BIGGER_SIZE = 16

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=TITLE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXES_LABEL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize

color = fg('blue')

if dataset == 'pgnorta':
    train = np.load('dataset/pgnorta/train_pgnorta.npy')
    test = np.load('dataset/pgnorta/test_pgnorta.npy')
    count_WGAN_npz_path = 'evaluate/results/pgnorta/samples.npz'
elif dataset == 'pgnorta_gan':
    train = np.load('dataset/pgnorta/train_pgnorta.npy')
    test = np.load('dataset/pgnorta/test_pgnorta.npy')
    count_WGAN_npz_path = 'evaluate/results/pgnorta_gan/samples.npz'


N_TRAIN = np.shape(train)[0]
count_WGAN = np.load(count_WGAN_npz_path)['count_WGAN']


# marginal mean & var, past & future correlation, W-distance for PGnorta & DS-WGAN, with CI
n_rep_CI = count_WGAN.shape[0] # how many replications to get the CI
n_sample = count_WGAN.shape[1] # how many samples generated for each replication, should equal to N_TRAIN
P = count_WGAN.shape[2] # how many time intervals, should equal to P

result_dir = 'evaluate/results/' + dataset + '/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


marginal_mean_DSWGAN_rec = np.zeros((n_rep_CI, P))
marginal_var_DSWGAN_rec = np.zeros((n_rep_CI, P))
past_future_corr_DSWGAN_rec = np.zeros((n_rep_CI, P-1))
# past_future_corr_test_rec = np.zeros((n_rep_CI, P-1))
past_future_corr_train = get_corr(train)
past_future_corr_test = get_corr(test)


print(color + 'Now computing CI' + attr('reset'))
for i in progressbar.progressbar(range(n_rep_CI)):
    marginal_mean_DSWGAN_rec[i,:], marginal_var_DSWGAN_rec[i,:] = np.mean(count_WGAN[i], axis=0), np.var(count_WGAN[i], axis=0)
    past_future_corr_DSWGAN_rec[i,:] = get_corr(count_WGAN[i])
    # past_future_corr_test_rec[i,:] = get_corr(test[i*n_sample:(i+1)*n_sample, :])
    



CI_percent = 95



    
marginal_mean_DSWGAN_CI = get_CI(marginal_mean_DSWGAN_rec, CI_percent, mode='quantile')
marginal_var_DSWGAN_CI = get_CI(marginal_var_DSWGAN_rec, CI_percent, mode='quantile')
past_future_corr_DSWGAN_CI = get_CI(past_future_corr_DSWGAN_rec, CI_percent, mode='quantile')


fill_between_alpha = 0.3
plt.figure(figsize=(6,4))
plt.plot(np.mean(test,axis=0), label='Test data set')
plt.plot(np.mean(train,axis=0), label='Training data set')
plt.fill_between(np.arange(P),marginal_mean_DSWGAN_CI['low'], marginal_mean_DSWGAN_CI['up'], label='CI given by DS-WGAN',alpha=fill_between_alpha)
plt.legend()
plt.xlabel('Time Interval')
plt.ylabel('Mean of arrival count')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
# plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
plt.tight_layout()
plt.savefig(result_dir + 'pgnorta_mean.pdf')
plt.close()

plt.figure(figsize=(6,4))
plt.plot(np.var(test,axis=0), label='Test data set')
plt.plot(np.var(train,axis=0), label='Training data set')
plt.fill_between(np.arange(P),marginal_var_DSWGAN_CI['low'], marginal_var_DSWGAN_CI['up'], label='CI given by DS-WGAN',alpha=fill_between_alpha)

plt.legend()
plt.xlabel('Time interval')
plt.ylabel('Variance of arrival count')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
plt.tight_layout()
plt.savefig(result_dir + 'pgnorta_var.pdf')
plt.close()

plt.figure(figsize=(6,4))
plt.plot(past_future_corr_test, label='Test data set')
plt.plot(past_future_corr_train, label='Training data set')

plt.fill_between(np.arange(P-1),past_future_corr_DSWGAN_CI['low'], past_future_corr_DSWGAN_CI['up'], label='CI given by DS-WGAN',alpha=fill_between_alpha)

plt.legend()
plt.xlabel('$j$')
plt.ylabel(r'$\operatorname{Corr}\left(\mathbf{Y}_{1: j}, \mathbf{Y}_{j+1: p}\right)$')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
# plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
plt.tight_layout()
plt.savefig(result_dir + 'pgnorta_corr.pdf')
plt.close('all')