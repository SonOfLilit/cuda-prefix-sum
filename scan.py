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
        try:
            result = scan_func(input)
        except:
            print(len(input), input)
            raise
        assert np.allclose(result, output), (len(input), input, result, output, output - result)
    print(scan_func.__name__, 'passed')

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
def scan_padding(x, n=None, out=None, block_i=0, block_dim=None, sums=None):
    if n is None:
        n = len(x)
    if block_dim is None:
        block_dim = n
    data = [-1] * (block_dim + block_dim // BANKS)
    assert n <= 2 * T
    threads = min((n + 1) // 2, T)

    for i in range(threads): # GPU similator
        if i < n:
            data[i + i // BANKS] = x[block_i * block_dim + i]
        k = i + (n + 1) // 2
        if k < n:
            data[k + k // BANKS] = x[block_i * block_dim + k]

    depth_power = 1
    while depth_power < n:
        for i in range(threads): # GPU similator
            ai = 2 * depth_power * (i + 1) - 1
            if ai < n:
                bi = ai - depth_power
                data[ai + ai // BANKS] += data[bi + bi // BANKS]
        depth_power *= 2

    depth_power //= 2
    while depth_power >= 1:
        for i in range(threads): # GPU similator
            ai = 2 * depth_power * (i + 1) - 1 + depth_power
            if ai < n:
                bi = ai - depth_power
                data[ai + ai // BANKS] += data[bi + bi // BANKS]
        depth_power //= 2

    if out is None:
        out = [-1] * n
    for i in range(threads): # GPU similator
        if i < n:
            out[block_i * block_dim + i] = data[i + i // BANKS]
        k = i + (n + 1) // 2
        if k < n:
            out[block_i * block_dim + k] = data[k + k // BANKS]

    for i in range(threads): # GPU similator
        if sums is not None and i == 0:
            sums[block_i] = out[block_i * block_dim + n - 1]

    return out
test(scan_padding)

def scan_large(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    block = 2 * T
    assert n < block * T
    blocks = (n + block - 1) // block
    sums = np.zeros(blocks, dtype=np.float32)

    for i in range(T): # GPU similator
        if i < blocks:
            scan_padding(x, n=min(block, n - i * block), out=x, block_i=i, block_dim=block, sums=sums)

    scan_padding(sums, out=sums)

    for i in range(T): # GPU similator
        if i < blocks - 1:
            i += 1
            x[block * i : min(block * (i + 1), n)] += sums[i - 1]

    return x

test_cases += [
  (np.ones(i), np.arange(1, i + 1))
  for i in [33, 63, 64, 65, 127, 128, 129, 256, 512, 1024]
]

test(scan_large)

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule(open("scan.cu").read())

scan_slow_cuda = mod.get_function("scan_slow")
scan_cuda = mod.get_function("scan")
scan_padding_cuda = mod.get_function("scan_padding")

copy_cuda = mod.get_function("copy")
def copy_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    copy_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=((n + 1) // 2, 1, 1), grid=(1,1), shared=4*n)
    return dest
for i in range(2048 + 1):
    assert np.allclose(copy_gpu(np.arange(i)), np.arange(i))

def scan_slow_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    scan_slow_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=(n, 1, 1), grid=(1,1), shared=2 * 4 * n)
    return dest

test(scan_slow_gpu)

def scan_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    scan_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=((n + 1) // 2, 1, 1), grid=(1,1), shared=4*n)
    return dest
test_cases += [
  (np.ones(i), np.arange(1, i + 1))
  for i in [1025, 2047, 2048]
]
test(scan_gpu)

def scan_padding_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    if n == 0:
        return dest
    scan_padding_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=((n + 1) // 2, 1, 1), grid=(1, 1), shared=4*(n + n // 16))
    return dest
test(scan_padding_gpu)

add_cuda = mod.get_function("add")
def stream_test(a):
    x = np.array(a, dtype=np.float32)
    n = len(x)
    if n == 0:
        return
    s = cuda.Stream()
    x_gpu = cuda.mem_alloc(x.nbytes)
    t_gpu = cuda.mem_alloc(x.nbytes)
    cuda.memcpy_htod_async(x_gpu, x, stream=s)
    cuda.memcpy_dtod_async(t_gpu, x_gpu, x.nbytes, stream=s)
    add_cuda(
        t_gpu, x_gpu, np.int32(n), stream=s,
        block=((n + 1) // 2, 1, 1), grid=(1,1))
    add_cuda(
        x_gpu, t_gpu, np.int32(n), stream=s,
        block=((n + 1) // 2, 1, 1), grid=(1,1))
    add_cuda(
        t_gpu, x_gpu, np.int32(n), stream=s,
        block=((n + 1) // 2, 1, 1), grid=(1,1))
    s.synchronize()
    cuda.memcpy_dtoh(x, t_gpu)
    assert np.allclose(x / 5.0, a), (a, x)
import itertools
for i in itertools.chain(range(32), range(33, 2048 + 1, 11)):
    stream_test(list(range(i)))
    stream_test(np.random.random_sample(i).astype(np.float32))

a = np.random.random_sample(2048).astype(np.float32)
print((scan_padding_gpu(a) - a.cumsum()).max())
