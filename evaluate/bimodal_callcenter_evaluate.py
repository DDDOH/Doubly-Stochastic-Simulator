from utils import estimate_PGnorta, evaluate_marginal, get_CI, evaluate_joint
from colored import fg, attr
import numpy as np
import progressbar
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_label', type=str, default='bikeshare', choices=[
    'infinite_server_queue',
    'bimodal', 'bimodal_gan',
    'callcenter_0', 'callcenter_0.5', 'callcenter_1', 'callcenter_2', 'callcenter_2.5',
    'callcenter_gan_0', 'callcenter_gan_0.5', 'callcenter_gan_1', 'callcenter_gan_2', 'callcenter_gan_2.5',
    'bikeshare'])


parser.add_argument('--reproduce', type=bool, default=True)
# if REPRODUCE, load the training dataset from the folder /reproduce
# else, load the training dataset from the folder /dataset
# only works for bimodal dataset


args = parser.parse_args()
dataset = args.exp_label
REPRODUCE = args.reproduce


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

DEBUG = False
# if debug, when estimate PGnorta model use few iterations
if DEBUG:
    ITERS = 300
    SAVE_FREQ = 100
    MAX_T = 10
else:
    ITERS = 30000
    SAVE_FREQ = 1000
    MAX_T = 4000

MARGIN = dataset
if dataset in ['bimodal', 'bimodal_gan']:
    if REPRODUCE:
        train = np.load('reproduce/reproduce_margin_bimodal__P_16/train_bimodal.npy')
    else:
        train = np.load('dataset/margin_bimodal__P_16/train_bimodal.npy')
    count_WGAN_npz_path = 'evaluate/results/{}/samples.npz'.format(dataset)
    mode = 'quantile'
    mode_var = mode
elif 'callcenter' in dataset:
    if 'gan' in dataset:
        # remove '_gan' from dataset
        training_set_name = dataset.split('_')[0] + '_' + dataset.split('_')[2]
    else:
        training_set_name = dataset
    train = np.load('dataset/callcenter/{}.npy'.format(training_set_name))
    count_WGAN_npz_path = os.path.join('evaluate/results', dataset, 'samples.npz')
    mode = 'quantile'
    mode_var = 'normal'
elif dataset == 'pgnorta':
    train = np.load('dataset/pgnorta/train_pgnorta.npy')
    count_WGAN_npz_path = 'evaluate/results/pgnorta/samples.npz'
    mode = 'quantile'
    mode_var = mode
# elif dataset == 'pgnorta_gan':
#     train = np.load('dataset/pgnorta/train_pgnorta.npy')
#     count_WGAN_npz_path = 'evaluate/results/pgnorta_gan/samples.npz'
elif dataset == 'bikeshare':
    mode = 'quantile'
    mode_var = mode
    train = np.load('dataset/bikeshare/training_set.npy')
    count_WGAN_npz_path = 'evaluate/results/bikeshare/samples.npz'


N_TRAIN = np.shape(train)[0]
count_WGAN = np.load(count_WGAN_npz_path)['count_WGAN']


# marginal mean & var, past & future correlation, W-distance for PGnorta & DS-WGAN, with CI
n_rep_CI = count_WGAN.shape[0] # how many replications to get the CI
n_sample = count_WGAN.shape[1] # how many samples generated for each replication, should equal to N_TRAIN
P = count_WGAN.shape[2] # how many time intervals, should equal to P

result_dir = 'evaluate/results/' + dataset + '/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# get count_PGnorta_mat
if DEBUG:
    norta = estimate_PGnorta(train, zeta=7/16, max_T=MAX_T, M=100, img_dir_name=result_dir+'sumamry_PGnorta_rho_estimation_record.jpg', rho_mat_dir_name=result_dir+'rho_mat.npy')
else:
    norta = estimate_PGnorta(train, zeta=7/16, max_T=MAX_T, M=100, img_dir_name=result_dir+'sumamry_PGnorta_rho_estimation_record.jpg', rho_mat_dir_name=result_dir+'rho_mat.npy')

marginal_mean_PGnorta_rec = np.zeros((n_rep_CI, P))
marginal_var_PGnorta_rec = np.zeros((n_rep_CI, P))
marginal_mean_DSWGAN_rec = np.zeros((n_rep_CI, P))
marginal_var_DSWGAN_rec = np.zeros((n_rep_CI, P))
past_future_corr_PGnorta_rec = np.zeros((n_rep_CI, P-1))
past_future_corr_DSWGAN_rec = np.zeros((n_rep_CI, P-1))
past_future_corr_train = None
W_distance_PGnorta_rec = np.zeros((n_rep_CI, P))
W_distance_DSWGAN_rec = np.zeros((n_rep_CI, P))


print(color + 'Now computing CI' + attr('reset'))
for i in progressbar.progressbar(range(n_rep_CI)):
    marginal_mean_DSWGAN_rec[i,:], marginal_var_DSWGAN_rec[i,:] = np.mean(count_WGAN[i], axis=0), np.var(count_WGAN[i], axis=0)

    count_PGnorta_mat = norta.sample_count(n_sample)
    marginal_mean_PGnorta_rec[i,:], marginal_var_PGnorta_rec[i,:] = np.mean(count_PGnorta_mat, axis=0), np.var(count_PGnorta_mat, axis=0)


    w_dist = evaluate_marginal(count_WGAN[i], count_PGnorta_mat, train, result_dir)
    past_future_corr = evaluate_joint(count_WGAN[i], count_PGnorta_mat, train, result_dir)

    W_distance_PGnorta_rec[i,:] = w_dist['PG']
    W_distance_DSWGAN_rec[i,:] = w_dist['WGAN']
    past_future_corr_PGnorta_rec[i,:] = past_future_corr['PG']
    past_future_corr_DSWGAN_rec[i,:] = past_future_corr['WGAN']
    past_future_corr_train = past_future_corr['TRAIN']

CI_percent = 95

marginal_mean_PGnorta_CI = get_CI(marginal_mean_PGnorta_rec, CI_percent, mode=mode)
marginal_var_PGnorta_CI = get_CI(marginal_var_PGnorta_rec, CI_percent, mode=mode_var)
marginal_mean_DSWGAN_CI = get_CI(marginal_mean_DSWGAN_rec, CI_percent, mode=mode)
marginal_var_DSWGAN_CI = get_CI(marginal_var_DSWGAN_rec, CI_percent, mode=mode_var)
past_future_corr_PGnorta_CI = get_CI(past_future_corr_PGnorta_rec, CI_percent, mode=mode)
past_future_corr_DSWGAN_CI = get_CI(past_future_corr_DSWGAN_rec, CI_percent, mode=mode)
W_distance_PGnorta_CI = get_CI(W_distance_PGnorta_rec, CI_percent, mode=mode)
W_distance_DSWGAN_CI = get_CI(W_distance_DSWGAN_rec, CI_percent, mode=mode)


fill_between_alpha = 0.3
plt.figure()
plt.fill_between(np.arange(P),marginal_mean_PGnorta_CI['low'], marginal_mean_PGnorta_CI['up'], label='PGnorta',alpha=fill_between_alpha)
plt.fill_between(np.arange(P),marginal_mean_DSWGAN_CI['low'], marginal_mean_DSWGAN_CI['up'], label='DS-WGAN',alpha=fill_between_alpha)
plt.plot(np.mean(train,axis=0), label='Training set',c='C2')
plt.legend()
plt.xlabel('Time Interval')
plt.ylabel('Mean of arrival count')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
# plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
plt.tight_layout()
plt.savefig(result_dir + '{}_compare_mean.pdf'.format(MARGIN))

plt.figure()
plt.fill_between(np.arange(P),marginal_var_PGnorta_CI['low'], marginal_var_PGnorta_CI['up'], label='PGnorta',alpha=fill_between_alpha)
plt.fill_between(np.arange(P),marginal_var_DSWGAN_CI['low'], marginal_var_DSWGAN_CI['up'], label='DS-WGAN',alpha=fill_between_alpha)
plt.plot(np.var(train,axis=0), label='Training set',c='C2')
plt.legend()
plt.xlabel('Time interval')
plt.ylabel('Variance of arrival count')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
plt.tight_layout()
plt.savefig(result_dir + '{}_compare_var.pdf'.format(MARGIN))

plt.figure()
plt.fill_between(np.arange(P-1),past_future_corr_PGnorta_CI['low'], past_future_corr_PGnorta_CI['up'], label='PGnorta',alpha=fill_between_alpha)
plt.fill_between(np.arange(P-1),past_future_corr_DSWGAN_CI['low'], past_future_corr_DSWGAN_CI['up'], label='DS-WGAN',alpha=fill_between_alpha)
plt.plot(past_future_corr_train, label='Training set',c='C2')
plt.legend()
plt.xlabel('$j$')
plt.ylabel(r'$\operatorname{Corr}\left(\mathbf{Y}_{1: j}, \mathbf{Y}_{j+1: p}\right)$')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
plt.tight_layout()
plt.savefig(result_dir + '{}_compare_past_future_corr.pdf'.format(MARGIN))
plt.close('all')

plt.figure()
plt.fill_between(np.arange(P),W_distance_PGnorta_CI['low'], W_distance_PGnorta_CI['up'], label=r'$D_{j}^{(P)}$',alpha=fill_between_alpha)
plt.fill_between(np.arange(P),W_distance_DSWGAN_CI['low'], W_distance_DSWGAN_CI['up'], label=r'$D_{j}^{(D)}$',alpha=fill_between_alpha)
plt.legend()
plt.xlabel('Time Interval $j$')
plt.ylabel('Wasserstein distance')
if P < 20:
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
else:
    plt.xticks(ticks=np.arange(0, P, 2), labels=np.arange(0, P, 2)+1)
plt.tight_layout()
plt.savefig(result_dir + '{}_compare_w_dist.pdf'.format(MARGIN))
plt.close('all')





# arrival count mat for ecdf and histogram
n_sample = 10000

# flatten the first two dimensions of count_WGAN
# 23-3-13 the better way is to load the saved generator network
count_WGAN_mat = count_WGAN.reshape(-1, count_WGAN.shape[-1])
assert count_WGAN_mat.shape[0] >= n_sample
count_WGAN_mat = count_WGAN_mat[:n_sample, :]
count_PGnorta_mat = norta.sample_count(n_sample)

# plot cdf & ecdf, calculate statistics for marginal distribution
for interval in progressbar.progressbar(range(P)):
    plt.figure()
    ecdf_count_PGnorta = sns.ecdfplot(
        data=count_PGnorta_mat[:,interval], alpha=0.3, label='PGnorta')
    ecdf_count_WGAN = sns.ecdfplot(
        data=count_WGAN_mat[:, interval], alpha=0.3, label='DS-WGAN')
    ecdf_train = sns.ecdfplot(
        data=train[:, interval], alpha=0.3, label='Training set')
    plt.legend()
    plt.xlabel('Arrival count')
    if interval == 0:
        plt.title('Empirical c.d.f of marginal arrival count for 1-st time interval'.format(interval+1))
    if interval == 1:
        plt.title('Empirical c.d.f of marginal arrival count for 2-nd time interval'.format(interval+1))
    if interval == 2:
        plt.title('Empirical c.d.f of marginal arrival count for 3-rd time interval'.format(interval+1))
    else:
        plt.title('Empirical c.d.f of marginal arrival count for {}-th  time interval'.format(interval+1))
    plt.tight_layout()

    plt.savefig(result_dir + '{}_marginal_count_ecdf'.format(MARGIN) +
                str(interval) + '.pdf')
    plt.close()

    plt.figure()
    bins = 50
    fig_alpha = 0.2
    plt.hist(count_PGnorta_mat[:,interval], bins=bins, alpha=fig_alpha,
                label='PGnorta', density=True)
    plt.hist(count_WGAN_mat[:, interval], bins=bins, alpha=fig_alpha,
                label='DS-WGAN', density=True)
    plt.hist(train[:, interval], bins=bins, alpha=fig_alpha,
                label='Training set', density=True)
    plt.xlabel('Arrival count')
    if interval == 0:
        plt.title('Histogram of marginal arrival count\nfor 1-st time interval'.format(interval+1))
    if interval == 1:
        plt.title('Histogram of marginal arrival count\nfor 2-nd time interval'.format(interval+1))
    if interval == 2:
        plt.title('Histogram of marginal arrival count\nfor 3-rd time interval'.format(interval+1))
    else:
        plt.title('Histogram of marginal arrival count\nfor {}-th time interval'.format(interval+1))
    sns_data = {'PGnorta': count_PGnorta_mat[:,interval],
                'DS-WGAN': count_WGAN_mat[:, interval],
                'Training': train[:, interval]}
    sns.kdeplot(data=sns_data, common_norm=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir + '{}_marginal_count_hist'.format(MARGIN) +
                str(interval) + '.pdf')
    plt.close()




