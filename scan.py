import numpy as np
import numpy.typing as npt

test_nums = [3, 1, 7, 0, 4, 1, 6, 3]
# 32 is the largest size that works for GPU scans that need 2x shared mem
random_test_nums = np.random.random_sample(32).astype(np.float32)
def cumsum(x):
    result = []
    s = 0
    for a in x:
        s += a
        result.append(s)
    return result
assert cumsum([3, 2, 1]) == [3, 5, 6]
assert cumsum([3, 1, 7]) == [3, 4, 11]

T = 32
def scan_slow(x, n):
    assert len(x) == n
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
assert scan_slow(test_nums, len(test_nums)) == cumsum(test_nums)
assert np.allclose(scan_slow(random_test_nums, len(random_test_nums)), cumsum(random_test_nums))

f = lambda x, depth_power: 2*depth_power*(x + 1) - 1
ff = lambda depth_power: [f(x, depth_power) for x in range(4)]
assert ff(1) == [1, 3, 5, 7], ff(1)
assert ff(2) == [3, 7, 11, 15], ff(2)
f = lambda x, depth_power: 2*depth_power*(x + 1) - 1 - depth_power
assert ff(1) == [0, 2, 4, 6], ff(1)
assert ff(2) == [1, 5, 9, 13], ff(2)

def scan(x, n):
    data = list(x)
    depth_power = 1
    while depth_power < n:
        for i in range(T): # GPU similator
            if depth_power * i < n // 2:
                data[2 * depth_power * (i + 1) - 1] += data[2 * depth_power * (i + 1) - 1 - depth_power]
        depth_power *= 2

    depth_power //= 2
    while depth_power >= 1:
        for i in range(T): # GPU similator
            if depth_power * (i + 1) < n // 2:
                data[2 * depth_power * (i + 1) - 1 + depth_power] += data[2 * depth_power * (i + 1) - 1]
        depth_power //= 2
    return data
assert scan(test_nums, len(test_nums)) == cumsum(test_nums)
assert np.allclose(scan(random_test_nums, len(random_test_nums)), cumsum(random_test_nums))

BANKS = 4
def scan_padding(x, n):
    data = [-1] * (T * BANKS)
    
    for i in range(T): # GPU similator
        if i < n:
            data[i + i // BANKS] = x[i]

    depth_power = 1
    while depth_power < n:
        for i in range(T): # GPU similator
            if depth_power * i < n // 2:
                ai = 2 * depth_power * (i + 1) - 1
                bi = 2 * depth_power * (i + 1) - 1 - depth_power
                data[ai + ai // BANKS] += data[bi + bi // BANKS]
        depth_power *= 2

    depth_power //= 2
    while depth_power >= 1:
        for i in range(T): # GPU similator
            if depth_power <= depth_power * (i + 1) < n // 2:
                ai = 2 * depth_power * (i + 1) - 1 + depth_power
                bi = 2 * depth_power * (i + 1) - 1
                data[ai + ai // BANKS] += data[bi + bi // BANKS]
        depth_power //= 2

    out = [-1] * n
    for i in range(T): # GPU similator
        if i < n:
            out[i] = data[i + i // BANKS]

    return out
assert scan_padding(test_nums, len(test_nums)) == cumsum(test_nums)
assert np.allclose(scan_padding(random_test_nums, len(random_test_nums)), cumsum(random_test_nums))

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void scan_slow(float *g_odata, float *g_idata, const int n) {
    extern __shared__ float data[];
    int i = threadIdx.x;
    // double buffering indexes
    int a = 0, b = 1;
    data[b * n + i] = g_idata[i];
    __syncthreads();
    for (int depth_power = 1; depth_power < n; depth_power *= 2) {
        // swap double buffer indices
        a = 1 - a;
        b = 1 - b;

        if (i >= depth_power) {
            data[b * n + i] = data[a * n + i] + data[a * n + i - depth_power];
        } else {
            data[b * n + i] = data[a * n + i];
        }
        __syncthreads();
    }
    g_odata[i] = data[b * n + i];
}

__global__ void scan(float *g_odata, float *g_idata, const int n) {
    extern __shared__ float data[];
    int i = threadIdx.x;
    data[2 * i] = g_idata[2 * i];
    data[2 * i + 1] = g_idata[2 * i + 1];
    __syncthreads();
    int depth_power = 1;
    for (; depth_power < n; depth_power <<= 1) {
        if (depth_power * i < n / 2) {
            data[2 * depth_power * (i + 1) - 1] += data[2 * depth_power * (i + 1) - 1 - depth_power];
        }
        __syncthreads();
    }
    for (depth_power >>= 1; depth_power >= 1; depth_power >>= 1) {
        if (depth_power * (i + 1) <= n / 2) {
            data[2 * depth_power * (i + 1) - 1 + depth_power] += data[2 * depth_power * (i + 1) - 1];
        }
        __syncthreads();
    }
    g_odata[2 * i] = data[2 * i];
    g_odata[2 * i + 1] = data[2 * i + 1];
}

__global__ void scan_padding(float *g_odata, float *g_idata, int n) {
    extern __shared__ float data[];
}""")

scan_slow_cuda = mod.get_function("scan_slow")
scan_cuda = mod.get_function("scan")
scan_padding_cuda = mod.get_function("scan_padding")

def scan_slow_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    scan_slow_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=(n,1,1), grid=(1,1), shared=2*n)
    return dest
result = scan_slow_gpu(test_nums)
assert np.allclose(result, cumsum(test_nums)), result
assert np.allclose(scan_slow_gpu(random_test_nums), cumsum(random_test_nums))

def scan_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    scan_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=(n // 2, 1, 1), grid=(1,1), shared=n)
    return dest
result = scan_gpu(test_nums)
assert np.allclose(result, cumsum(test_nums)), result
assert np.allclose(scan_gpu(random_test_nums), cumsum(random_test_nums))

def scan_padding_gpu(data: npt.ArrayLike) -> npt.NDArray[np.float32]:
    x = np.array(data, dtype=np.float32)
    dest = np.zeros_like(x)
    n = len(x)
    scan_padding_cuda(
        cuda.Out(dest), cuda.In(x), np.int32(n),
        block=(n // 2,1,1), grid=(1,1), shared=2*n)
    return dest
result = scan_padding_gpu(test_nums)
assert np.allclose(result, cumsum(test_nums)), result
assert np.allclose(scan_padding_gpu(random_test_nums), cumsum(random_test_nums))

a = np.random.random_sample(32).astype(np.float32)
print((scan_gpu(a) - a.cumsum()))
