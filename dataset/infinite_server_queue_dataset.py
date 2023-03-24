########## For Figure 4, infinite server queue ##########
import math

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from scipy.interpolate import interp1d
from scipy.stats import gamma

class ArrivalProcess():
    def __init__(self, T, arrival_ls):
        self.arrival_ls = arrival_ls
        self.T = T
        self.n_arrival = len(arrival_ls)
    def run_through():
        return 1
    
    def set_service_time(self, service_ls):
        self.service_ls = service_ls
    
    def get_count_vector(self, interval_len):
        return np.histogram(self.arrival_ls, bins=np.arange(0, self.T + interval_len, interval_len))[0]


def simulate_CIR_arrival():
    Z_ls = simulate_CIR()
    max_lam = np.max(Z_ls)
    N_arrival = np.random.poisson(max_lam * T)
    unfiltered_arrival_time = np.sort(np.random.uniform(0, T, size=N_arrival))

    keep_prob = np.array([Z_ls[int(t/dt)] for t in unfiltered_arrival_time])/max_lam
    whether_keep = np.random.rand(len(keep_prob)) <= keep_prob

    filtered_arrival_time = unfiltered_arrival_time[whether_keep]
    return filtered_arrival_time


def simulate_CIR():
    sqrt_dt = np.sqrt(dt)
    beta = 100
    B_t = np.random.normal(size=N)
    
    Z_t_0 = R_t_on_T_ls[0] * gamma.rvs(a=beta, scale=1/beta)
    Z_ls = np.zeros(N)
    Z_ls[0] = Z_t_0
    for i in range(N - 1):
        t = T_ls[i]
        Z_t = Z_ls[i]
        R_t_val = R_t_on_T_ls[i]
        R_t_exp_alpha_val = R_t_exp_alpha_on_T_ls[i]

        
        d_Z_t = kappa * (R_t_val - Z_t) * dt + sigma * R_t_exp_alpha_val * (Z_t**0.5) * sqrt_dt * B_t[i]
        Z_ls[i+1] = Z_t + d_Z_t
    return Z_ls

def infinite_server_queue(arrival_ls, sampler, eval_t_ls):
    # calculate number of occupied servers at time t
    # given arrival time and service time for each customer
    service_ls = sampler(size=len(arrival_ls))
    if (type(eval_t_ls) is int) or (type(eval_t_ls) is float):
        eval_t_ls = np.array(eval_t_ls)
    end_ls = arrival_ls + service_ls
    return np.array([np.sum((end_ls >= t) & (arrival_ls <= t)) for t in eval_t_ls])

def arrival_epoch_sampler(count_vector):
    total_arrival = np.sum(count_vector)
    arrival_ls = np.zeros(total_arrival)
    index = 0
    for i in range(P):
        interval_start = i * interval_len
        interval_end = interval_start + interval_len
        arrival_ls_one_interval = np.random.uniform(interval_start, interval_end, size=count_vector[i])
        arrival_ls[index:index+count_vector[i]] = arrival_ls_one_interval
        index += count_vector[i]
    arrival_ls = np.sort(arrival_ls)
    return arrival_ls

# $\mathrm{d} \lambda(t)=\kappa(R(t)-\lambda(t)) \mathrm{d} t+\sigma R(t)^{\alpha} \lambda(t)^{1 / 2} \mathrm{~d} B(t)$

# infinite server queue
kappa = 0.2
sigma = 0.4
alpha = 0.3

# service time distribution
lognormal_var = 0.1
lognormal_mean = 0.2
normal_sigma = (math.log(lognormal_var / lognormal_mean ** 2 + 1))**0.5
normal_mean = math.log(lognormal_mean) - normal_sigma ** 2 / 2

# the service time sampler
sampler = lambda size: np.random.lognormal(mean=normal_mean,sigma=normal_sigma,size=size)
service_rate = 1/np.mean(sampler(1000))

eval_t_ls = np.arange(0,11,0.05)

P = 22
interval_len = 0.5
T = P * interval_len


if __name__ == '__main__':

    # figure font size setting
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


    

    base_lam = np.array([124, 175, 239, 263, 285,
                    299, 292, 276, 249, 257,
                    274, 273, 268, 259, 251,
                    252, 244, 219, 176, 156,
                    135, 120])
    x = np.linspace(interval_len/2, T-interval_len/2, len(base_lam))
    y = base_lam
    f2 = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
    def R_t(t):
        assert t >= 0
        assert t <= T
        return f2(t)
    R_t = np.vectorize(R_t)

    R_i = []
    for i in range(P):
        interval_start = i * interval_len
        interval_end = (i + 1) * interval_len
        t_ls = np.arange(interval_start, interval_end, 0.002)
        r_ls = R_t(t_ls)
        R_i.append(np.mean(r_ls))
    R_i = np.array(R_i)




    # multi server queue
    # kappa = 3
    # # sigma = 0.4
    # sigma = 2
    # alpha = 0.3
            
    # 计算 R(t) 在每个离散的仿真点上的数值，所有CIR都用这个
    N = 5000
    T_ls = np.linspace(0, T, N, endpoint=False)
    R_t_on_T_ls = R_t(T_ls)
    R_t_exp_alpha_on_T_ls = R_t_on_T_ls ** alpha
    dt = T/N


    plt.figure(figsize=(6,4))
    plt.plot(T_ls, R_t_on_T_ls)
    plt.xlabel('$t$')
    plt.xticks(np.arange(11.01))
    plt.ylabel('$R(t)$')
    plt.tight_layout()
    plt.savefig('dataset/infinite_server_queue/R_t.pdf')


    # get real CIR arrival process, for plotting the true line
    real_CIR_size = 3000
    real_CIR_ls = np.ndarray((real_CIR_size,),dtype=object)
    real_count_mat = np.zeros((real_CIR_size, P))
    for i in progressbar.progressbar(range(real_CIR_size)):
        real_CIR_ls[i] = ArrivalProcess(T=T,arrival_ls=simulate_CIR_arrival())
        real_count_mat[i,:] = real_CIR_ls[i].get_count_vector(interval_len)

    # get training CIR arrival count matrix
    training_CIR_size = 300
    training_CIR_ls = np.ndarray((training_CIR_size,),dtype=object)
    training_count_mat = np.zeros((training_CIR_size, P))
    for i in progressbar.progressbar(range(training_CIR_size)):
        training_CIR_ls[i] = ArrivalProcess(T=T,arrival_ls=simulate_CIR_arrival())
        training_count_mat[i,:] = training_CIR_ls[i].get_count_vector(interval_len)

    np.save('dataset/infinite_server_queue/training_set.npy', training_count_mat)


    # run simulation for real CIR arrival process


    real_count_mat = real_count_mat.astype(int)
    real_PC_size = len(real_count_mat)
    real_PC_ls = np.ndarray((real_PC_size,),dtype=object)
    for i in progressbar.progressbar(range(real_PC_size)):
        real_PC_ls[i] = ArrivalProcess(T=T,arrival_ls=arrival_epoch_sampler(real_count_mat[i,:]))

    
    # PC result is not reported anymore
    # real_PC_n_occupied = np.zeros((real_PC_size, len(eval_t_ls)))
    # for i in progressbar.progressbar(range(real_PC_size)):
    #     real_PC_n_occupied[i,:] = infinite_server_queue(real_PC_ls[i].arrival_ls, sampler, eval_t_ls)

    real_CIR_n_occupied = np.zeros((real_CIR_size, len(eval_t_ls)))
    for i in progressbar.progressbar(range(real_CIR_size)):
        real_CIR_n_occupied[i,:] = infinite_server_queue(real_CIR_ls[i].arrival_ls, sampler, eval_t_ls)


    # save simulation result for real CIR arrival process
    np.savez('dataset/infinite_server_queue/real_CIR_n_occupied.npz',
                real_CIR_n_occupied=real_CIR_n_occupied)

