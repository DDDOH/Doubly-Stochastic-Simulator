import numpy as np
from numpy import linalg as la
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt
from evaluate.utils import get_corr


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
    

# ********** The code to find the nearset positive definite matrix ********** #
# Reference: https://stackoverflow.com/a/43244194


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3



def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    

def get_PGnorata_from_img():
    """Get parameters for PGnorta model from the image.

    Returns:
        (PGnorta): A PGnorta instance with the parameters come from images.
    """

    # read correlation matrix.
    corr_img = 'dataset/pgnorta/2_commercial_call_center_corr.png'
    img = mpimg.imread(corr_img)
    if corr_img == 'dataset/pgnorta/2_commercial_call_center_corr.png':
        colorbar = img[:, 1255, :3]
        p = 22
        width = 1220
        height = 1115

    n_color = np.shape(colorbar)[0]
    width_interval = width / (p + 1)
    height_interval = height/(p+1)
    width_loc = np.arange(width_interval/2, width-width_interval/2, p)
    height_loc = np.arange(height_interval/2, height-height_interval/2, p)
    corr_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            rgb_ij = img[int(height_loc[i]), int(width_loc[j]), :3]
            color_dist = np.sum(np.abs(colorbar - rgb_ij), axis=1)
            corr_mat[i, j] = 1 - np.argmin(color_dist)/n_color
    corr_mat = nearestPD(corr_mat)

    # read mean curve.
    mean_img = 'dataset/pgnorta/2_commercial_call_center_mean.png'
    img = mpimg.imread(mean_img)
    line_img = img[28:625, 145:1235, :3]
    count_min = 100
    count_max = 350
    y_axis_length, x_axis_length = np.shape(line_img)[0], np.shape(line_img)[1]

    loc = np.linspace(0, x_axis_length-1, p)
    mean = np.zeros(p)
    for i in range(p):
        mean[i] = (1 - np.argmin(line_img[:, int(loc[i]), 1]) /
                   y_axis_length)*(count_max-count_min) + count_min


    # DI data is from Figure 8, the length of DI is exactly p.
    DI = np.array([4.9079754601226995,
        6.226993865030675,
        7.024539877300614,
        7.085889570552148,
        6.411042944785277,

        7.883435582822086,
        8.25153374233129,
        7.668711656441718,
        6.779141104294479,
        7.914110429447853,

        9.386503067484663,
        8.957055214723926,
        9.110429447852761,
        8.036809815950921,
        8.742331288343559,

        9.785276073619631,
        10.0920245398773,
        10.306748466257668,
        8.650306748466257,
        8.374233128834355,

        7.975460122699387,
        7.116564417177914])

    var = DI * mean
    alpha = mean ** 2 / (var - mean)
    return PGnorta(base_intensity=mean, cov=corr_mat, alpha=alpha)



# data = get_PGnorata_from_img()


N = 22
base_lam = np.array([124, 175, 239, 263, 285,
                     299, 292, 276, 249, 257,
                     274, 273, 268, 259, 251,
                     252, 244, 219, 176, 156,
                     135, 120])


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
I = mpimg.imread('dataset/pgnorta/matshow.png')
# print(I.shape)
# plt.imshow(I)
bar = mpimg.imread('dataset/pgnorta/colorbar.png')
x_loc_ls = []
y_loc_ls = []
corr_mat = np.zeros((22,22))
for i in range(22):
    for j in range(22):
        y_loc = int(668 / 22 * j + 668/22/2)
        x_loc = int(610 / 22 * i + 610/22/2)
        x_loc_ls.append(x_loc)
        y_loc_ls.append(y_loc)
        color = I[x_loc, y_loc,:3]
        corr_mat[i,j] = 1 - np.argmin(np.apply_along_axis(lambda x: np.linalg.norm(x), 1, color-bar[:,10,:3]))/605
# plt.scatter(y_loc_ls, x_loc_ls,s=2)

for i in range(22):
    corr_mat[i,i] = 1
corr_mat = (corr_mat + corr_mat.T)/2
# plt.matshow(corr_mat)
# plt.colorbar()


DI = np.array([5.1, 6.4, 7.2, 7.3, 6.6,
               8.0, 8.4, 7.8, 7.0, 8.1,
               9.5, 9.1, 9.2, 8.2, 8.9,
               9.9, 10.2, 10.4, 8.8, 8.5,
               8.1, 7.3])

var_b_j = (DI - 1)/base_lam
alpha=1/var_b_j

cov_mat = np.array(corr_mat)
alpha = np.array(alpha)
base_lam = np.array(base_lam)
# n_known = len(count)

data = PGnorta(base_intensity=base_lam, cov=cov_mat, alpha=alpha)


seq_len = data.seq_len
training_intensity, training_count = data.sample_both(n_sample=300)
test_intensity, test_count = data.sample_both(n_sample=20000)
# real_intensity = data.sample_intensity(n_sample=2000)
# training_set = torch.tensor(training_count, dtype=torch.float)


test_corr = get_corr(test_count)
train_corr = get_corr(training_count)

plt.plot(test_corr, label='test')
plt.plot(train_corr, label='train')
plt.legend()
plt.savefig('dataset/pgnorta/corr.png')
plt.close()


plt.plot(np.mean(training_count, axis=0), label='training')
plt.plot(np.mean(test_count, axis=0), label='test')
plt.legend()
plt.savefig('dataset/pgnorta/mean.png')
plt.close()


plt.plot(np.var(training_count, axis=0), label='training')
plt.plot(np.var(test_count, axis=0), label='test')
plt.legend()
plt.savefig('dataset/pgnorta/var.png')
plt.close()


np.save('dataset/pgnorta/train_pgnorta.npy', training_count)
np.save('dataset/pgnorta/test_pgnorta.npy', test_count)

