from http.client import UnimplementedFileMode
import progressbar
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import os
from colored import fg, attr

class PGnorta():
    def __init__(self, base_intensity, cov, alpha):
        """Initialize a PGnorta dataset loader.

        Args:
            base_intensity (np.array): A list containing the mean of arrival count in each time step.
            cov (np.array): Covariance matrix of the underlying normal copula.
            alpha (np.array): A list containing the parameter of gamma distribution in each time step.
        """
        assert len(base_intensity) == len(alpha) and len(alpha) == np.shape(
            cov)[0] and np.shape(cov)[0] == np.shape(cov)[1]
        assert min(base_intensity) > 0, 'only accept nonnegative intensity'
        self.base_intensity = base_intensity
        self.p = len(base_intensity)  # the sequence length
        self.cov = cov
        self.alpha = alpha
        self.seq_len = len(base_intensity)

    def z_to_lam(self, Z, first=True):
        """Convert Z to intensity.

        Args:
            Z (np.array): The value of the normal copula for the first or last severl time steps or the whole sequence.
                          For one sample, it can be either a one dimension list with length q, or a 1 * q array.
                          For multiple samples, it should be a n * q array, n is the number of samples.
            first (bool, optional): Whether the given Z is for the first several time steps or not. Defaults to True.

        Returns:
            intensity (np.array): The value of intensity suggested by Z. An array of the same shape as Z.
        """
        if Z.ndim == 1:
            n_step = len(Z)
        else:
            n_step = np.shape(Z)[1]
        U = norm.cdf(Z)
        if first:
            B = gamma.ppf(q=U, a=self.alpha[:n_step],
                          scale=1/self.alpha[:n_step])
            intensity = B * self.base_intensity[:n_step]
        else:
            B = gamma.ppf(q=U, a=self.alpha[-n_step:],
                          scale=1/self.alpha[-n_step:])
            intensity = B * self.base_intensity[-n_step:]
        return intensity
    
    def sample_intensity(self, n_sample):
        """Sample arrival intensity from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            intensity (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        z = multivariate_normal.rvs(np.zeros(self.p), self.cov, n_sample)
        intensity = self.z_to_lam(z)
        return intensity
        

    def sample_count(self, n_sample):
        """Sample arrival count from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            count (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        intensity = self.sample_intensity(n_sample)
        count = np.random.poisson(intensity)
        return count
    
    def sample_both(self, n_sample):
        """Sample both arrival count and intensity from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            count (np.array): An array of size (n_sample * seq_len), each row is one sample.
            intensity (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        intensity = self.sample_intensity(n_sample)
        count = np.random.poisson(intensity)
        return intensity, count



def estimate_PGnorta(count_mat, zeta=9/16, max_T=1000, M=100, img_dir_name=None, rho_mat_dir_name=None):
    p = np.shape(count_mat)[1]
    lam = np.mean(count_mat, axis=0)
    var_X = np.var(count_mat, axis=0)
    alpha = lam ** 2 / (var_X - lam)

    if np.min(alpha) < 0:
        print('The arrival count of the {}-th time interval does not satisfy variance >= mean'.format(np.where(alpha < 0)[0]))
        
    alpha[alpha < 0] = 10000 # alpha 越大，则生成的arrival count的mean和variance越接近

    kappa_t = lambda t : 0.1 * t ** (- zeta)
    rho_jk_record = np.zeros((p,p,max_T))


    # if tile rho_mat_dir_name exist, read it directly
    if os.path.exists(rho_mat_dir_name):
        print(fg('blue') + 'Loading rho_matrix directly.' + attr('reset'))
        rho_jk_record = np.load(rho_mat_dir_name)
    else:
        print(fg('blue') + 'No existing rho_matrix file. Estimate the model now.' + attr('reset'))
        with progressbar.ProgressBar(max_value=p ** 2) as bar:
            n_estimated = 0
            for j in range(p):
                for k in range(p):
                    if j == k:
                        rho_jk_record[j,k,:] = 1
                        continue
                    rho_jk = 0
                    hat_r_jk_X = spearmanr(count_mat[:,j], count_mat[:,k])[0]
                    for t in range(1, max_T):
                        # for m = 1 to M do
                        Z = multivariate_normal.rvs(np.zeros(2), [[1,rho_jk],[rho_jk,1]], M)
                        U = norm.cdf(Z)
                        B_j = gamma.ppf(q=U[:,0], a=alpha[j], scale=1/alpha[j])
                        B_k = gamma.ppf(q=U[:,1], a=alpha[k], scale=1/alpha[k])
                        T_j, T_k = lam[j] * B_j, lam[k] * B_k
                        X_j, X_k = np.random.poisson(T_j), np.random.poisson(T_k)
                        # end for

                        tilde_r_jk_X = spearmanr(X_j, X_k)[0]

                        rho_jk += kappa_t(t) * (hat_r_jk_X - tilde_r_jk_X)
                        rho_jk_record[j,k,t] = rho_jk
                    # plt.figure()
                    # plt.plot(rho_jk_record[j,k,:])
                    # plt.show()
                    # plt.close()
                    n_estimated += 1
                    if img_dir_name is not None:
                        n_plot = 0
                        plt.figure()
                        for j_ in range(p):
                            for k_ in range(p):
                                if rho_jk_record[j_, k_, -1] != 0: 
                                    plt.plot(rho_jk_record[j_,k_,:])
                                    n_plot += 1
                                    if n_plot == 50:
                                        break
                        plt.title('rho estimation trajectory')
                        plt.savefig(img_dir_name)
                        plt.close()
                    bar.update(n_estimated)
        if rho_mat_dir_name is not None:
            np.save(rho_mat_dir_name, rho_jk_record)
            print(fg('blue') + 'rho_matrix saved to file.' + attr('reset'))
    norta = PGnorta(base_intensity=lam, cov=rho_jk_record[:,:,-1], alpha=alpha)
    return norta



def sample_PGnorta_marginal(base_intensity_t, alpha_t, n_sample):
    z = np.random.normal(0, 1, n_sample)
    U = norm.cdf(z)
    B = gamma.ppf(q=U, a=alpha_t,
                  scale=1/alpha_t)
    intensity = B * base_intensity_t
    count = np.random.poisson(intensity)
    return intensity, count



import numpy as np
import scipy
import matplotlib.pyplot as plt


def evaluate_marginal(count_WGAN, count_PGnorta, count_train, result_dir):
    assert np.shape(count_WGAN)[1] == np.shape(
        count_PGnorta)[1] == np.shape(count_train)[1]

    wasserstein_PG_train_rec = []
    wasserstein_WGAN_train_rec = []
    p = np.shape(count_WGAN)[1]
    for interval in range(p):
        wasserstein_PG_train_rec.append(scipy.stats.wasserstein_distance(
            count_PGnorta[:, interval], count_train[:, interval]))
        wasserstein_WGAN_train_rec.append(scipy.stats.wasserstein_distance(
            count_WGAN[:, interval], count_train[:, interval]))

    return {'PG': wasserstein_PG_train_rec, 'WGAN': wasserstein_WGAN_train_rec}

    # plt.figure()
    # plt.plot(wasserstein_PG_train_rec, label='PGnorta & Train')
    # plt.plot(wasserstein_WGAN_train_rec, label='DS-WGAN & Train')
    # plt.title('Wasserstein distance')
    # plt.legend()
    # plt.savefig(result_dir + 'summary_wasserstein_distance.png')
    # plt.close()


def get_corr(count_mat):
    cumsum_count = np.cumsum(count_mat, axis=1)
    sum_count = np.sum(count_mat, axis=1)
    p = np.shape(count_mat)[1]
    past_future_corr = []
    for interval in range(p - 1):
        past_future_corr.append(scipy.stats.spearmanr(
            cumsum_count[:, interval], sum_count - cumsum_count[:, interval])[0])
    return np.array(past_future_corr)

def evaluate_joint(count_WGAN, count_PGnorta, count_train, result_dir):
    assert np.shape(count_WGAN)[1] == np.shape(
        count_PGnorta)[1] == np.shape(count_train)[1]

    # past_future_corr_WGAN = []
    # past_future_corr_PG = []
    # past_future_corr_train = []

    # cumsum_WGAN = np.cumsum(count_WGAN, axis=1)
    # cumsum_PG = np.cumsum(count_PGnorta, axis=1)
    # cumsum_train = np.cumsum(count_train, axis=1)

    # sum_WGAN = np.sum(count_WGAN, axis=1)
    # sum_PG = np.sum(count_PGnorta, axis=1)
    # sum_train = np.sum(count_train, axis=1)

    # p = np.shape(count_WGAN)[1]
    # for interval in range(p - 1):
    #     past_future_corr_WGAN.append(scipy.stats.pearsonr(
    #         cumsum_WGAN[:, interval], sum_WGAN - cumsum_WGAN[:, interval])[0])
    #     past_future_corr_PG.append(scipy.stats.pearsonr(
    #         cumsum_PG[:, interval], sum_PG - cumsum_PG[:, interval])[0])
    #     past_future_corr_train.append(scipy.stats.pearsonr(
    #         cumsum_train[:, interval], sum_train - cumsum_train[:, interval])[0])

    past_future_corr_PG = get_corr(count_PGnorta)
    past_future_corr_WGAN = get_corr(count_WGAN)
    past_future_corr_train = get_corr(count_train)
    return {'PG': past_future_corr_PG, 'WGAN': past_future_corr_WGAN, 'TRAIN': past_future_corr_train}


def compare_plot(real, fake, PGnorta_mean, PGnorta_var, msg, result_dir, save=False):
    """Visualize and compare the real and fake.
    """
    real_size = np.shape(real)[0]
    fake_size = np.shape(fake)[0]

    P = np.shape(real)[1]

    assert np.shape(real)[1] == np.shape(
        fake)[1] == len(PGnorta_mean) == len(PGnorta_var)

    max_intensity = max(np.max(fake), np.max(real))
    plt.figure(figsize=(16, 3))
    plt.subplot(141)
    plt.plot(np.mean(fake, axis=0), label='fake')
    plt.plot(np.mean(real, axis=0), label='real')
    plt.plot(PGnorta_mean, label='PGnorta')
    plt.xlabel('Time interval')
    plt.ylabel('Intensity or count')
    plt.title('Mean')
    plt.legend()

    plt.subplot(142)
    plt.plot(np.var(fake, axis=0), label='fake')
    plt.plot(np.var(real, axis=0), label='real')
    plt.plot(PGnorta_var, label='PGnorta')
    plt.xlabel('Time interval')
    plt.title('Std')
    plt.legend()

    plt.subplot(143)
    plt.scatter(np.tile(np.arange(P), real_size).reshape(
        P, real_size), real, alpha=0.003)
    plt.ylim(0, max_intensity * 1.2)
    plt.xlabel('Time interval')
    plt.title('Scatter plot of real')

    plt.subplot(144)
    plt.scatter(np.tile(np.arange(P), fake_size).reshape(
        P, fake_size), fake, alpha=0.003)
    plt.ylim(0, max_intensity * 1.2)
    plt.xlabel('Time interval')
    plt.title('Scatter plot of fake')
    if save:
        plt.savefig(
            result_dir + msg + '.png')
    else:
        plt.show()
    plt.close()


def get_CI(data_mat, percent, mode='normal'):
    """Get confidence interval for multi-dimension data.

    Args:
        data_mat (_type_): np.array of shape (n_rep, n_dim).
        percent (_type_): in range (0,100)
        mode: normal or quantile
    """
    if mode == 'normal':
        n_sample = np.shape(data_mat)[0]
        mean = np.mean(data_mat, axis=0)
        std = np.std(data_mat, axis=0)
        return {'up': mean + std * scipy.stats.norm.ppf(1 - (100 - percent) / 200) / np.sqrt(n_sample),
                'low': mean + std * scipy.stats.norm.ppf((100 - percent) / 200) / np.sqrt(n_sample)}
    elif mode == 'quantile':
        return {'up': np.quantile(data_mat, 1 - (100 - percent) / 200, axis=0),
                'low': np.quantile(data_mat, (100 - percent) / 200, axis=0)}

    


if __name__ == '__main__':
    data_mat = np.random.randn(200, 10)
    CI = get_CI(data_mat, 95)


def get_marginal_wass(count_WGAN, count_PGnorta, count_train):
    assert np.shape(count_WGAN)[1] == np.shape(
        count_PGnorta)[1] == np.shape(count_train)[1]

    wasserstein_PG_train_rec = []
    wasserstein_WGAN_train_rec = []
    p = np.shape(count_WGAN)[1]
    for interval in range(p):
        wasserstein_PG_train_rec.append(scipy.stats.wasserstein_distance(
            count_PGnorta[:, interval], count_train[:, interval]))
        wasserstein_WGAN_train_rec.append(scipy.stats.wasserstein_distance(
            count_WGAN[:, interval], count_train[:, interval]))
    return {'PG': wasserstein_PG_train_rec, 'WGAN': wasserstein_WGAN_train_rec}