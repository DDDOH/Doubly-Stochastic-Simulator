# run this file with python -m evaluate.infinite_server_queue.py

import numpy as np
import matplotlib.pyplot as plt
import progressbar
from dataset.infinite_server_queue_dataset import ArrivalProcess, arrival_epoch_sampler, sampler, eval_t_ls, infinite_server_queue, T
from .utils import get_CI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=str, default='False')
# if debug, run smaller n_rep
args = parser.parse_args()
debug = args.debug == 'True'

# TODO 23-3-14 Load network instead
npz_path = 'evaluate/results/infinite_server_queue/samples.npz'
fake_count_mat = np.load(npz_path)['count_WGAN']
# count_WGAN, pred_intensity

if debug:
    fake_count_mat = fake_count_mat[:20,:,:]
    
TRAIN_SIZE = fake_count_mat.shape[1]
n_rep = fake_count_mat.shape[0]
P = fake_count_mat.shape[2]

# reshape fake_count_mat to (3 * 300, 22)
fake_count_mat = fake_count_mat.reshape((n_rep * TRAIN_SIZE, P))
fake_count_mat.shape


fake_count_mat = fake_count_mat.astype(int)
fake_PC_size = TRAIN_SIZE * n_rep
fake_PC_ls = np.ndarray((fake_PC_size,),dtype=np.object)
for i in progressbar.progressbar(range(fake_PC_size)):
    fake_PC_ls[i] = ArrivalProcess(T=T, arrival_ls=arrival_epoch_sampler(fake_count_mat[i,:]))

fake_PC_n_occupied = np.zeros((fake_PC_size, len(eval_t_ls)))
for i in progressbar.progressbar(range(fake_PC_size)):
    fake_PC_n_occupied[i,:] = infinite_server_queue(fake_PC_ls[i].arrival_ls, sampler, eval_t_ls) 



fake_n_occupied_mean_mat = np.zeros((n_rep, len(eval_t_ls)))
fake_n_occupied_var_mat = np.zeros((n_rep, len(eval_t_ls)))
for i in range(n_rep):
    start_id = i * TRAIN_SIZE
    end_id = (i + 1) * TRAIN_SIZE
    fake_n_occupied_mean_one_rep = np.mean(fake_PC_n_occupied[start_id:end_id,:],axis=0)
    fake_n_occupied_var_one_rep = np.var(fake_PC_n_occupied[start_id:end_id,:],axis=0)
    fake_n_occupied_mean_mat[i,:] = fake_n_occupied_mean_one_rep
    fake_n_occupied_var_mat[i,:] = fake_n_occupied_var_one_rep

fake_mean_CI = get_CI(fake_n_occupied_mean_mat, percent=95, mode='quantile')
fake_var_CI = get_CI(fake_n_occupied_var_mat, percent=95, mode='quantile')


real_CIR_n_occupied = np.load('dataset/infinite_server_queue/real_CIR_n_occupied.npz')['real_CIR_n_occupied']

plt.figure(figsize=(6,4))

plt.plot(eval_t_ls, np.mean(real_CIR_n_occupied,axis=0),label='True')
plt.fill_between(eval_t_ls, fake_mean_CI['low'], fake_mean_CI['up'], alpha=0.4, label='CI generated by DS-WGAN and\nrun-through-queue experiments')
plt.xlabel('$t$')
plt.xticks(np.arange(11.01))
plt.ylabel('Mean of number\nof occupied servers')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('evaluate/results/infinite_server_queue/CIR_mean.pdf')

# plt.plot(eval_t_ls, np.mean(real_PC_n_occupied,axis=0),label='real PC')
plt.figure(figsize=(6,4))
plt.plot(eval_t_ls, np.var(real_CIR_n_occupied,axis=0),label='True')
plt.fill_between(eval_t_ls, fake_var_CI['low'], fake_var_CI['up'], alpha=0.4, label='CI generated by DS-WGAN and\nrun-through-queue experiments')
plt.xlabel('$t$')
plt.xticks(np.arange(11.01))
plt.ylabel('Variance of number\nof occupied servers')
# plt.plot(eval_t_ls, np.var(real_PC_n_occupied,axis=0),label='real PC')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('evaluate/results/infinite_server_queue/CIR_var.pdf')

