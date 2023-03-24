# test gamma distribution parameters and mean, var, std
from scipy.stats import gamma
# we need gamma distribution of mean 1 and var 1/alpha
alpha = 0.2
scale = 1/alpha
mean = alpha * scale
var = alpha * scale**2
dist = gamma(a=alpha, scale=scale)
print("mean: ", dist.mean())
print("var: ", dist.var())
print("std: ", dist.std())