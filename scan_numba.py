import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "1"; os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

import numpy as np
from numba import cuda

TPB = 256
BANKS = 16
SHARED_SIZE = 2 * TPB + 2 * TPB // BANKS
@cuda.jit(device=True)
def scan_block_kernel(dest, src, n, shared):
    block_pos = 2 * cuda.blockDim.x * cuda.blockIdx.x
    n = min(2 * TPB, n - block_pos) # from now on, it is block-local n
    t = cuda.threadIdx.x
    half_n = n // 2
    if t > half_n:
        return

    pos = block_pos + t

    t_padded = t + t // BANKS
    shared[t_padded] = src[pos]
    t_plus_half = t + half_n
    t_plus_half_padded = t_plus_half + t_plus_half // BANKS
    if t_plus_half < n:
        shared[t_plus_half_padded] = src[pos + half_n]
    cuda.syncthreads()

    d = 1
    while d < n:
        a = 2 * d * (t + 1) - 1
        if a < n:
            b = a - d
            shared[a + a // BANKS] += shared[b + b // BANKS]
        d *= 2
        cuda.syncthreads()

    d //= 2
    while d >= 1:
        a = 2 * d * (t + 1) - 1 + d
        if a < n:
            b = a - d
            shared[a + a // BANKS] += shared[b + b // BANKS]
        d //= 2
        cuda.syncthreads()

    dest[pos] = shared[t_padded]
    if t_plus_half < n:
        dest[pos + half_n] = shared[t_plus_half_padded]


@cuda.jit
def scan_kernel(dest, src, n, sums):
    block_pos = 2 * cuda.blockDim.x * cuda.blockIdx.x
    block_n = min(2 * TPB, n - block_pos) # from now on, it is block-local n
    t = cuda.threadIdx.x
    half_block_n = block_n // 2
    if t > half_block_n:
        return
    pos = block_pos + t
    t_padded = t + t // BANKS
    t_plus_half = t + half_block_n


    shared = cuda.shared.array(shape=SHARED_SIZE, dtype=src.dtype)

    scan_block_kernel(dest, src, n, shared)

    if cuda.threadIdx.x == 0 and cuda.blockIdx.x < cuda.gridDim.x - 1:
        print(cuda.gridDim, cuda.blockDim, cuda.blockIdx, cuda.threadIdx)
        sums[cuda.blockIdx.x] = dest[2 * cuda.blockDim.x * (cuda.blockIdx.x + 1) - 1]

    cuda.threadfence_block()

    if cuda.blockIdx.x == 0:
        return

    if cuda.blockIdx.x == 1:
        scan_block_kernel(sums, sums, cuda.gridDim.x - 1, shared)
    cuda.threadfence_block()

    prev_block_sum = sums[cuda.blockIdx.x - 1]
    dest[pos] += prev_block_sum
    if t_plus_half < block_n:
        dest[pos + half_block_n] += prev_block_sum

def idiv_round_up(x, y):
    return (x + y - 1) // y

def scan(x):
    n, = x.shape
    sums_device = cuda.device_array(shape=idiv_round_up(n, TPB), dtype=x.dtype)
    x_device = cuda.to_device(x)
    result_device = cuda.device_array_like(x)
    blocks = idiv_round_up(n, 2 * TPB)
    print('launching', n, TPB, blocks)
    scan_kernel[blocks, TPB](result_device, x_device, n, sums_device)
    result = result_device.copy_to_host()
    return result

for dtype in [np.int32, np.float32]:
    for i in [1, 2, 3, 8, 10, 20, 32, 256, 1023, 1024, 1025, 2048, 1024 * 1024, 1024 * 1024 - 1, 1024 * 1024 + 1]:
        print(dtype, i)
        x = np.ones(i, dtype=dtype)
        s = scan(x)
        assert np.allclose(s, np.arange(1, i + 1)), (i, x)
