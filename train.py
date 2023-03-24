import torch.optim as optim
from geomloss import SamplesLoss
import os
from models.poisson_simulator import DSSimulator
import numpy as np
import torch
import progressbar
import matplotlib.pyplot as plt
from utils import create_result_folder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_label', type=str, default='bikeshare')
# infinite_server_queue: for figure 4
# pgnorta: figure 6
# bimodal
# callcenter callcenter_1 callcenter_0.5
# bikeshare: for figure 10, 11

parser.add_argument('--reproduce', type=bool, default=True)
# weather use the same training set as the paper.
# Only influence bimodal, since the training set sampled can be quite different from the paper if resample.
# if True, load the data from the folder reproduce
# if False, load the data from the folder dataset
# exp_label = infinite_server_queue does not need REPRODUCE since using randomly sampled training data has little effect.
# exp_label = callcenter does not need REPRODUCE since we have a fixed dataset.



args = parser.parse_args()
exp_label = args.exp_label

REPRODUCE = args.reproduce

result_dir = '/Users/shuffleofficial/Offline_Documents/Doubly_Stochastic_WGAN/tmp_results/'
result_dir = create_result_folder(result_dir, exp_label)
os.mkdir(os.path.join(result_dir, 'models'))


'''
# mode: sinkhorn or discriminator
# sinkhorn means we use the geomloss package. 
# discriminator means we use the discriminator network.
# sinkhorn is much more faster but performance is not as good as discriminator.
'''

if exp_label == 'infinite_server_queue':
    data_path = 'dataset/infinite_server_queue/training_set.npy'
    training_set = np.load(data_path)
    n_repetitions = 100
    dim = [512, 512]
    mode = 'sinkhorn'
    batch_size = None
    seed_dim = 11
elif exp_label == 'pgnorta':
    data_path = 'dataset/pgnorta/train_pgnorta.npy'
    training_set = np.load(data_path)
    n_repetitions = 100
    dim = [512, 512, 512, 512, 256]
    mode = 'sinkhorn'
    batch_size = None
    seed_dim = 4
elif exp_label == 'bimodal':
    if REPRODUCE:
        data_path = 'reproduce/reproduce_margin_bimodal__P_16/train_bimodal.npy'
    else:
        data_path = 'dataset/margin_bimodal__P_16/train_bimodal.npy'
    training_set = np.load(data_path)
    n_repetitions = 100
    dim = [512, 512]
    mode = 'sinkhorn'
    batch_size = None
    seed_dim = 11
elif 'callcenter' in exp_label:
    data_path = 'dataset/callcenter/{}.npy'.format(exp_label)
    training_set = np.load(data_path)
    n_repetitions = 100
    dim = [256, 256]
    batch_size = 256
    mode = 'discriminator'
    seed_dim = 11
elif exp_label == 'bikeshare':
    data_path = 'dataset/bikeshare/training_set.npy'
    training_set = np.load(data_path)
    n_repetitions = 100
    mode = 'sinkhorn'
    dim = [512, 512, 512, 512, 256]
    batch_size = None
    seed_dim = 11


training_set = torch.from_numpy(training_set).float()
TRAIN_SIZE = np.shape(training_set)[0]
    
if batch_size is None:
    batch_size = TRAIN_SIZE
else:
    batch_size = batch_size


P = training_set.shape[1]


# initialize Doubly stochastic simulator
simulator = DSSimulator(seed_dim=seed_dim, hidden_size=dim,
                        n_interval=P)
sinkorn_loss = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.5)



lr_initial = 0.001
lr_rate = 10
iters = 30000
betas = (0.5, 0.9)
assert lr_rate >= 1, 'lr_final = lr_initial / lr_rate, thus lr_rate >= 1'
lr_final = lr_initial / lr_rate
gamma_G = (lr_final/lr_initial)**(1/iters)
optimizerG = optim.Adam(
    simulator.parameters(), lr=lr_initial, betas=betas)
optimizerG_lrdecay = torch.optim.lr_scheduler.ExponentialLR(
    optimizerG, gamma=gamma_G, last_epoch=-1)

G_cost_record = []
lr_record = []

save_freq = 1000
for iteration in progressbar.progressbar(range(iters), redirect_stdout=True):

    noise = torch.randn(batch_size, seed_dim)
    count_WGAN, pred_intensity = simulator(noise, return_intensity=True)
    # sinkorn_loss input shoud be two batch of samples of shape (batch_size, vector_dim)
    # randomly select batch_size rows from training_set
    idx = np.random.choice(TRAIN_SIZE, batch_size, replace=False)
    training_set_batch = training_set[idx, :]
    G_cost = sinkorn_loss(count_WGAN, training_set_batch)
    simulator.zero_grad()
    G_cost.backward()
    G_cost_record.append(G_cost.detach().cpu().numpy())
    optimizerG.step()
    optimizerG_lrdecay.step()

    lr_record.append(optimizerG_lrdecay.get_last_lr())
    
    if iteration % save_freq == 0 and iteration != 0:
        print('DS-Simulator loss in {}-th iteration: {}'.format(iteration, G_cost.item()))
        plt.figure()
        plt.semilogy(G_cost_record)
        plt.title('G_cost')
        plt.savefig(os.path.join(result_dir, 'G_cost.png'))
        plt.close()
        
        plt.figure()
        plt.plot(lr_record)
        plt.title('LR')
        plt.savefig(os.path.join(result_dir, 'LR_record.png'))
        plt.close()

        
        count_WGAN_ls = []
        pred_intensity_ls = []
        for i in range(n_repetitions):
            noise = torch.randn(TRAIN_SIZE, seed_dim)
            count_WGAN, pred_intensity = simulator(noise, return_intensity=True)
            count_WGAN_ls.append(count_WGAN.cpu().detach().numpy())
            pred_intensity_ls.append(pred_intensity.cpu().detach().numpy())
        np.savez(os.path.join(result_dir, 'samples_{}.npz'.format(iteration)), count_WGAN=count_WGAN_ls, pred_intensity=pred_intensity_ls)
        torch.save(simulator.state_dict(), os.path.join(
            result_dir, 'models', 'simulator_{}.pth'.format(iteration)))
        
        if not os.path.exists(os.path.join('evaluate/results', exp_label)):
            os.mkdir(os.path.join('evaluate/results', exp_label))

        # also save the latest model into evaluate/results/exp_label folder
        torch.save(simulator.state_dict(), os.path.join(
            'evaluate/results', exp_label, 'simulator.pth'))
        np.savez(os.path.join('evaluate/results', exp_label, 'samples.npz'), count_WGAN=count_WGAN_ls, pred_intensity=pred_intensity_ls)
