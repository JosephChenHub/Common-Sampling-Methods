from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

N = int(1e3 * 1e3 * 10)

# pi
X  = np.random.uniform(0., 1., (N, 2))

def within_circle(a, b):
    return a*a + b*b < 1

def criteria_exp(a, b):
    return a * b < 1

def count(x, lo, hi, criteria):
    res = 0
    for i in tqdm(range(lo, hi)):
        a, b = x[i]
        if criteria(a, b):
            res += 1
    return res

def error(a, b):
    return abs(a - b) / (1e-6 + b)

p = Pool()
num_process = 10
seg_len = N // num_process

result = []
for i in range(num_process):
    lo = seg_len * i
    hi = seg_len * (i+1)
    if i == num_process-1:
        hi = N
    result.append(p.apply_async(count, args=(X, lo, hi, within_circle)))

p.close()
p.join()

val = 0
for i in result:
    cnt = i.get()
    val += cnt

pi_ = val * 4.0 / X.shape[0]
print("Sampling:%s Approximate pi is %s " %(N, pi_),  "err:{}%".format(error(pi_, np.pi) * 100))

# e
result = []
X = np.random.uniform(1., 2., (N, 1))
Y = np.random.uniform(0., 1., (N, 1))

p = Pool()
for i in range(num_process):
    lo = seg_len * i
    hi = seg_len * (i+1)
    if i == num_process - 1:
        hi = N
    result.append(p.apply_async(count, args=(np.concatenate((X, Y), axis=1), lo, hi, criteria_exp)))
p.close()
p.join()

val = 0
for i in result:
    val += i.get()

exp = np.power(2, N * 1.0 / (1e-6 + val))
print("Sampling:%s Approximate exp is %s " %(N, exp),  "err:{}%".format(error(exp, np.exp(1)) * 100))

