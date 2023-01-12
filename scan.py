import numpy as np
import numpy.typing as npt

test_nums = np.array([3, 1, 7, 0, 4, 1, 6, 3])
# 32 is the largest size that works for GPU scans that need 2x shared mem
random_test_nums = np.random.random_sample(32).astype(np.float32)
test_cases = [
    (np.array([0]), np.array([0])),
    (np.array([1, 2, 3]), np.array([1, 3, 6])),
    (test_nums, test_nums.cumsum()),
    (np.arange(32), np.arange(32).cumsum()),
    (np.ones(32), np.arange(1, 33)),
    (random_test_nums, random_test_nums.cumsum()),
] + [(np.ones(i), np.arange(1, i + 1)) for i in range(32)]
def test(scan_func):
    for input, output in test_cases:
        result = scan_func(input)
        assert np.allclose(result, output), (input, result, output, output - result)

T = 32
def scan_slow(x):
    n = len(x)
    data = [list(x), [None] * n]
    depth_power = 1
    # double buffering
    a = 0
    b = 1
    while depth_power < n:
        for i in range(T): # gpu simulator
            if i < n:
                if i >= depth_power:
                    data[b][i] = data[a][i] + data[a][i - depth_power]
                else:
                    data[b][i] = data[a][i]
        depth_power *= 2
        a, b = b, a
    return data[a]
test(scan_slow)

f = lambda x, depth_power: 2*depth_power*(x + 1) - 1
ff = lambda depth_power: [f(x, depth_power) for x in range(4)]
assert ff(1) == [1, 3, 5, 7], ff(1)
assert ff(2) == [3, 7, 11, 15], ff(2)
f = lambda x, depth_power: 2*depth_power*(x + 1) - 1 - depth_power
assert ff(1) == [0, 2, 4, 6], ff(1)
assert ff(2) == [1, 5, 9, 13], ff(2)

def scan(x):
    n = len(x)
    data = list(x)
    depth_power = 1
    while depth_power < n:
        for i in range(T): # GPU similator
            offset = 2 * depth_power * (i + 1) - 1
            if offset < n:
                data[offset] += data[offset - depth_power]
        depth_power *= 2

    depth_power //= 2
    while depth_power >= 1:
        for i in range(T): # GPU similator
            offset = 2 * depth_power * (i + 1) - 1
            if offset + depth_power < n:
                data[offset + depth_power] += data[offset]
        depth_power //= 2
    return data
test(scan)

BANKS = 4
def scan_padding(x):
    n = len(x)
    data = [-1] * (n + n // BANKS + 1)
    
    for i in range(T): # GPU similator
        if i < n:
            data[i + i // BANKS] = x[i]

    depth_power = 1
    while depth_power < n:
        for i in range(T): # GPU similator
            ai = 2 * depth_power * (i + 1) - 1
            if ai <= n:
                bi = 2 * depth_power * (i + 1) - 1 - depth_power
                data[ai + ai // BANKS] += data[bi + bi // BANKS]
        depth_power *= 2

    depth_power //= 2
    while depth_power >= 1:
        for i in range(T): # GPU similator
            ai = 2 * depth_power * (i + 1) - 1 + depth_power
            if ai <= n:
                bi = 2 * depth_power * (i + 1) - 1
                data[ai + ai // BANKS] += data[bi + bi // BANKS]
        depth_power //= 2

    out = [-1] * n
    for i in range(T): # GPU similator
        if i < n:
            out[i] = data[i + i // BANKS]

    return out
test(scan_padding)

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule(open("scan.cu").read())

scan_slow_cuda = mod.get_function("scan_slow")
scan_cuda = mod.get_function("scan")
scan_padding_cuda = mod.get_function("scan_padding")

def scan_slow_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    scan_slow_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=(max(n, 1), 1, 1), grid=(1,1), shared=2*n)
    return dest
result = scan_slow_gpu(test_nums)
test(scan_slow_gpu)

def scan_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    scan_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=((n + 1) // 2, 1, 1), grid=(1,1), shared=n)
    return dest
test(scan_gpu)

def scan_padding_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    scan_padding_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=((n + 1) // 2, 1, 1), grid=(1, 1), shared=n + n // 16)
    return dest
test(scan_padding_gpu)

a = np.random.random_sample(32).astype(np.float32)
print((scan_padding_gpu(a) - a.cumsum()))
