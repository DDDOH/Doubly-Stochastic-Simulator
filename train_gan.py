import argparse
import numpy as np
import progressbar
from utils import create_result_folder
import os, sys, getopt
sys.path.append(os.getcwd())
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--exp_label', type=str, default='pgnorta_gan', choices=[
    'callcenter_gan_0', 'callcenter_gan_0.5', 'callcenter_gan_1', 'callcenter_gan_2', 'callcenter_gan_2.5',
    'bimodal_gan',
    'pgnorta_gan'
])
parser.add_argument('--reproduce', type=str, default='True')
# if REPRODUCE, load the training dataset from the folder reproduce
# else, load the training dataset from the folder dataset
# only works for bimodal dataset

args = parser.parse_args()
exp_label = args.exp_label
REPRODUCE = args.reproduce == 'True'

# Change result_dir here.
result_dir = '/Users/shuffleofficial/Offline_Documents/Doubly_Stochastic_WGAN/tmp_results/'
result_dir = create_result_folder(result_dir, exp_label)

OUTPUT = 'intensity'
LAMBDA = 1 # Smaller lambda seems to help for toy tasks specifically
DROPOUT = False
LAMBDA_GEN = 0
NOTE = ''
CRITIC_ITERS = 5 # 5 originally  # How many critic iterations per generator iteration
DIM = 512  # Model dimensionality
C_GAN = (int(1) == 1)
ITERS = 50000

class Normalize():
    def __init__(self, ori_data, target_mean, target_std):
        self.ori_mean = np.mean(ori_data)
        self.ori_std = np.std(ori_data)
        self.ori_data = ori_data.copy()
        
        def rescale_func(data):
            data = data.copy()
            zero_mean_one_std_data = (data - self.ori_mean)/self.ori_std
            rescaled_data = zero_mean_one_std_data * target_std + target_mean
            return rescaled_data.copy()
        
        self.rescale_func = rescale_func
        
        self.rescaled_mean = target_mean
        self.rescaled_std = target_std
        self.rescaled_ori = self.rescale_func(self.ori_data)
        
    def get_rescaled_on_ori(self):
        # given target_mean, target_std, get the rescaled ori_data
        return self.rescaled_ori.copy()
    
    def get_rescale_on_new(self, new_data):
        new_data = new_data.copy()
        rescaled_new = self.rescale_func(new_data)
        return rescaled_new.copy()
    
    def rescale_on_new_para(self):
        rescale_para = {}
        rescale_para['multipler'] = 1/self.ori_std*self.rescaled_std
        rescale_para['adder'] = - self.ori_mean/self.ori_std*self.rescaled_std + self.rescaled_mean
        return rescale_para
    
    def get_ori(self):
        return self.ori_data.copy()


if 'callcenter' in exp_label:
    # remove '_gan' from exp_label
    data_name = exp_label.split('_')[0] + '_' + exp_label.split('_')[2]
    training_set = np.load('dataset/callcenter/{}.npy'.format(data_name))
elif exp_label == 'bimodal_gan':
    if REPRODUCE:
        training_set = np.load('reproduce/reproduce_margin_bimodal__P_16/train_bimodal.npy')
    else:
        training_set = np.load('dataset/margin_bimodal__P_16/train_bimodal.npy')
elif exp_label == 'pgnorta_gan':
    training_set = np.load('dataset/pgnorta/train_pgnorta.npy')
    

C_GAN = False
para = {}
para['P_KNOWN'] = 0
para['P'] = 16
para['TRAIN_SIZE'] = len(training_set)
para['MAGNITUDE'] = np.mean(training_set)
para['DI_VAL'] = np.mean(np.var(training_set, axis=0)/np.mean(training_set, axis=0))
para['NOTE'] = ''


if exp_label == 'pgnorta_gan':
    para['P'] = 22
        
para["DIM"] = DIM
para["OUTPUT"] = OUTPUT
para["C_GAN"] = C_GAN

KNOWN_MASK = np.array([x < para['P_KNOWN'] for x in range(para['P'])])
UNKNOWN_MASK = np.array([not x for x in KNOWN_MASK])
SEED_DIM = np.sum(UNKNOWN_MASK)
# BATCH_SIZE = min(256, int(para['TRAIN_SIZE']/2))  # Batch size
BATCH_SIZE = 256
use_cuda = bool(torch.cuda.device_count())
PARA_CMT = "TRAIN_SIZE: {}, P_KNOWN: {}, Magnitude:{}, DI:{}, OUTPUT:{}, GradientPenalty:{}, GeneratorPenalty:{}, DROPOUT:{}, DIM:{}, {}".format(para['TRAIN_SIZE'], 
                                                        para['P_KNOWN'], 
                                                        para['MAGNITUDE'],
                                                        para['DI_VAL'],
                                                        OUTPUT,
                                                        LAMBDA,
                                                        LAMBDA_GEN,
                                                        DROPOUT,
                                                        DIM,
                                                        para['NOTE'])



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        main = nn.Sequential(
            nn.Linear(SEED_DIM + sum(KNOWN_MASK), DIM), nn.LeakyReLU(0.1, True),
            nn.Linear(DIM, DIM), nn.LeakyReLU(0.1, True),
            nn.Linear(DIM, DIM), nn.LeakyReLU(0.1, True),
            nn.Linear(DIM, para['P'] - sum(KNOWN_MASK)), # generate the intensity at the unknown period
        )
        self.main = main
    def forward(self, noise):
        output = self.main(noise)
        return output

if not DROPOUT:
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            main = nn.Sequential(
                nn.Linear(para['P'], DIM), nn.LeakyReLU(0.1),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1),
                nn.Linear(DIM, 1),
            )
            self.main = main
        def forward(self, inputs):
            output = self.main(inputs)
            return output.view(-1)
else:
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            main = nn.Sequential(
                nn.Linear(para['P'], DIM), nn.LeakyReLU(0.1), nn.Dropout(0.1, True),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1), nn.Dropout(0.2, True),
                nn.Linear(DIM, DIM), nn.LeakyReLU(0.1), nn.Dropout(0.4, True),
                nn.Linear(DIM, 1),
            )
            self.main = main
        def forward(self, inputs):
            output = self.main(inputs)
            return output.view(-1)



def inf_train_iter(data_set):
    while True:
        select_id = np.random.choice(np.arange(len(data_set)),BATCH_SIZE,replace=True)
        yield data_set[select_id,:]


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data[torch.randperm(BATCH_SIZE),:])

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty 

def calc_generator_penalty(netG, real_data, fake_data):
    assert para['P_KNOWN'] >= 1
    if LAMBDA_GEN == 0:
        return 0
    
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data[torch.randperm(BATCH_SIZE),:])

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    noise = torch.randn(BATCH_SIZE, SEED_DIM)
    noise = noise.cuda() if use_cuda else noise

    noisev = autograd.Variable(noise)
    gen_input = torch.cat([noisev, interpolates[:, KNOWN_MASK]], 1)

    # interpolates[:para['P_KNOWN'], :]
    gene_interpolates = netG(gen_input)

    gradients = autograd.grad(outputs=gene_interpolates, inputs=gen_input,
                              grad_outputs=torch.ones(gene_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  gene_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0][:,-para['P_KNOWN']:][:,para['GP_DIM']]

    
    generator_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() * LAMBDA_GEN
    
    return generator_penalty


real_count_iter = inf_train_iter(training_set)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if OUTPUT == 'intensity' or OUTPUT == 'count':
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(3)
        if OUTPUT == 'b':
            m.weight.data.normal_(0.0, 0.01)
            m.bias.data.fill_(0.01)

    # useless
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(1)



netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

one = torch.tensor(1, dtype=torch.float)
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()



lr_final = 1e-6
lr_initial = 1e-4
gamma_G = (lr_final/lr_initial)**(1/ITERS)
gamma_D = (lr_final/lr_initial)**(1/ITERS/CRITIC_ITERS)

optimizerD = optim.Adam(netD.parameters(), lr=lr_initial, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr_initial, betas=(0.5, 0.9))
optimizerD_lrdecay = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=gamma_D, last_epoch=-1)
optimizerG_lrdecay = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=gamma_G, last_epoch=-1)



for iteration in progressbar.progressbar(range(ITERS)):
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):#CRITIC_ITERS):
        # real count for getting condition
        _count = real_count_iter.__next__()
        real_count = torch.Tensor(_count)
        
        if use_cuda:
            real_count = real_count.cuda()
        real_count_v = autograd.Variable(real_count)

        # train with real
        D_real = netD(real_count_v)
        D_real = D_real.mean()

        # train with fake
        # use a new batch of data to penalize discriminator
        _count = real_count_iter.__next__()
        real_count = torch.Tensor(_count)
        # use naive random noise
        noise = torch.randn(BATCH_SIZE, SEED_DIM)
        
        if use_cuda:
            noise = noise.cuda()
            real_count = real_count.cuda()
        with torch.no_grad(): # totally freeze netG
            noisev = autograd.Variable(noise)
            gen_input = torch.cat([noisev, real_count[:, KNOWN_MASK]], 1)
        
        if OUTPUT == 'intensity':
            intensity_pred = netG(gen_input)
            sign = torch.sign(intensity_pred)
            intensity_pred = intensity_pred * sign
            count_fake = autograd.Variable(torch.poisson(intensity_pred.float().data))
            count_fake = count_fake * sign
            
        if OUTPUT == 'count':
            count_fake = netG(gen_input)
            
        all_count = real_count
        all_count[:, UNKNOWN_MASK] = count_fake
        all_count = autograd.Variable(all_count)
        
        D_fake = netD(all_count)
        D_fake = D_fake.mean()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_count, all_count.data)

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        
        netD.zero_grad()
        D_cost.backward()
        optimizerD.step()
        optimizerD_lrdecay.step()


    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    
    _count = real_count_iter.__next__()
    real_count = torch.Tensor(_count)
    
    if use_cuda:
        real_count = real_count.cuda()
    real_count = autograd.Variable(real_count)

    noise = torch.randn(BATCH_SIZE, SEED_DIM)
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise)
    gen_input = torch.cat([noisev, real_count[:, KNOWN_MASK]], 1)
    
    if OUTPUT == 'intensity':
        intensity_pred = netG(gen_input)
        intensity_val = intensity_pred.cpu().data.numpy() if use_cuda else intensity_pred.data.numpy()
        sign = np.sign(intensity_val)
        intensity_val = intensity_val * sign
        count_val = np.random.poisson(intensity_val)
        count_val = count_val * sign
        
        w_mid = 1 + (count_val - intensity_val)/(2 * intensity_val) 
        w_mid = np.maximum(w_mid, 0.5)
        w_mid = np.minimum(w_mid, 1.5)
        b_mid = count_val - w_mid * intensity_val

        w_mid_tensor = autograd.Variable(torch.Tensor([w_mid]))
        b_mid_tensor = autograd.Variable(torch.Tensor([b_mid]))
                             
        if use_cuda:
            w_mid_tensor = w_mid_tensor.cuda()
            b_mid_tensor = b_mid_tensor.cuda()
            
        pred_fake = intensity_pred * w_mid_tensor + b_mid_tensor
        
    if OUTPUT == 'count':
        pred_fake = netG(gen_input)

    all_fake = real_count
    all_fake[:, UNKNOWN_MASK] = pred_fake
    
    G = netD(all_fake)
    G = G.mean()

    G_cost = -G
    
    netG.zero_grad()
    G_cost.backward()
    optimizerG.step()
    optimizerG_lrdecay.step()

    if iteration % 1000 == 0 and iteration != 0:
        count_WGAN_ls = []
        pred_intensity_ls = []
        for i in range(100):
            noise = torch.randn(training_set.shape[0], SEED_DIM)
            if use_cuda:
                noise = noise.cuda()
            noisev = autograd.Variable(noise)
            gen_input = noisev

            if OUTPUT == 'intensity':
                intensity_pred = netG(gen_input)
                intensity_val = intensity_pred.cpu().data.numpy() if use_cuda else intensity_pred.data.numpy()
                sign = np.sign(intensity_val)
                intensity_val = intensity_val * sign
                count_val = np.random.poisson(intensity_val)
                count_val = count_val * sign
                
                w_mid = 1 + (count_val - intensity_val)/(2 * intensity_val) 
                w_mid = np.maximum(w_mid, 0.5)
                w_mid = np.minimum(w_mid, 1.5)
                b_mid = count_val - w_mid * intensity_val

                w_mid_tensor = autograd.Variable(torch.Tensor([w_mid]))
                b_mid_tensor = autograd.Variable(torch.Tensor([b_mid]))
                                    
                if use_cuda:
                    w_mid_tensor = w_mid_tensor.cuda()
                    b_mid_tensor = b_mid_tensor.cuda()
                    
                pred_fake = intensity_pred * w_mid_tensor + b_mid_tensor
                
            if OUTPUT == 'count':
                pred_fake = netG(gen_input)
            
            count_WGAN_ls.append(pred_fake.cpu().detach().numpy()[0])
            pred_intensity_ls.append(intensity_pred.cpu().detach().numpy())
        np.savez(os.path.join(result_dir, 'samples_{}.npz'.format(iteration)), count_WGAN=count_WGAN_ls, pred_intensity=pred_intensity_ls)

        if not os.path.exists(os.path.join('evaluate/results', exp_label)):
            os.mkdir(os.path.join('evaluate/results', exp_label))
        np.savez(os.path.join('evaluate/results', exp_label, 'samples.npz'), count_WGAN=count_WGAN_ls, pred_intensity=pred_intensity_ls)
