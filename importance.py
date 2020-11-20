import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

mlp = nn.Sequential(
    nn.Linear(1, 128),
    nn.Sigmoid(),
    nn.Linear(128, 128),
    nn.Sigmoid(),
    nn.Linear(128, 1)
)

mlp.eval()



""" to estimate the expection of this f(x) """
def transform(x, min_, max_):
    n = x.shape[0]
    x_ = torch.from_numpy(x).float().view(n, 1)
    y = mlp(x_).detach().numpy().squeeze()

    y[y > max_] = max_
    y[y < min_] = min_
    return y


def normal_pdf(x, mu, sigma):
    y = 1.0/np.sqrt(2 * np.pi * sigma * sigma + 1e-6) * np.exp(-(x - mu)**2 / (1e-6 + 2*sigma**2))
    return y

def uniform_pdf(x, a, b):
    y = np.zeros_like(x)
    tmp = 1.0 / ( b - a + 1e-6)
    y[...] = tmp
    return y

def logistic_pdf(x, mu, s):
    assert s > 0
    tmp = -(x-mu) / s
    px = np.exp(tmp) / s / (1 + np.exp(tmp))**2
    return px

mu_0 = -10
sigma_0 = 0.5
A, B = -50, 50

N = int(1e3 * 1e3 )
min_ = -1
max_ = 1

x = np.random.normal(mu_0, sigma_0,  N) # distribution : p(x)

# if we can directly sample from the original distribution
mean1 = transform(x, min_, max_).mean()
print("Mean from gaussian:", mean1)

# otherwise
x1 = np.random.uniform(A, B,  N)
e = transform(x1, min_, max_) * normal_pdf(x1, mu_0, sigma_0) / uniform_pdf(x1, A, B)
mean2 = e.mean()
print("Mean from uniform:", mean2)




t0 = np.linspace(-50, 50, 10000)
t1 = normal_pdf(t0, mu_0, sigma_0)
t2  = uniform_pdf(t0, A, B)
t3 = transform(t0, min_, max_)
plt.plot(t0, t1, 'r' )
plt.plot(t0, t2, 'g')
plt.text(mu_0, t1.max()-0.1, "gaussian", c='r')
plt.text(0, t2.max(), "uniform", c='g')
plt.plot(t0, t3, 'b')
plt.text(0, t3.max(), "f(x)", c='b')
plt.grid(True)
plt.title("Importance Sampling")

plt.legend(['E[f(x)] from gaussian:%s'%mean1,
            'E[f(x)] from uniform:%s'%mean2])


plt.show()
