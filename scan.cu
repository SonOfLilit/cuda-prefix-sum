__global__ void scan_slow(float *g_odata, float *g_idata, const int n)
{
    extern __shared__ float data[];
    int i = threadIdx.x;
    // double buffering indexes
    int a = 0, b = 1;
    if (i < n)
    {
        data[b * n + i] = g_idata[i];
    }
    __syncthreads();
    for (int depth_power = 1; depth_power < n; depth_power *= 2)
    {
        // swap double buffer indices
        a = 1 - a;
        b = 1 - b;

        if (i >= depth_power)
        {
            data[b * n + i] = data[a * n + i] + data[a * n + i - depth_power];
        }
        else
        {
            data[b * n + i] = data[a * n + i];
        }
        __syncthreads();
    }
    g_odata[i] = data[b * n + i];
}

__global__ void copy(float *g_odata, float *g_idata, const int n)
{
    extern __shared__ float data[];

    int i = threadIdx.x;
    __syncthreads();
    data[2 * i] = g_idata[2 * i];
    if (2 * i + 1 < n)
    {
        data[2 * i + 1] = g_idata[2 * i + 1];
    }
    __syncthreads();

    g_odata[2 * i] = data[2 * i];
    if (2 * i + 1 < n)
    {
        g_odata[2 * i + 1] = data[2 * i + 1];
    }
}

__global__ void add(float *g_odata, float *g_idata, const int n)
{
    int i = threadIdx.x;
    g_odata[i] += g_idata[i];
    i += (n + 1) / 2;
    if (i < n)
    {
        g_odata[i] += g_idata[i];
    }
}

__global__ void scan(float *g_odata, float *g_idata, const int n)
{
    extern __shared__ float data[];
    int i = threadIdx.x;
    __syncthreads();
    if (2 * i < n)
    {
        data[2 * i] = g_idata[2 * i];
    }
    if (2 * i + 1 < n)
    {
        data[2 * i + 1] = g_idata[2 * i + 1];
    }
    __syncthreads();
    int depth_power = 1;
    for (; depth_power < n; depth_power <<= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset < n)
        {
            data[offset] += data[offset - depth_power];
        }
        __syncthreads();
    }
    for (depth_power >>= 1; depth_power >= 1; depth_power >>= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset + depth_power < n)
        {
            data[offset + depth_power] += data[offset];
        }
        __syncthreads();
    }
    if (2 * i < n)
    {
        g_odata[2 * i] = data[2 * i];
        g_odata[2 * i + 1] = data[2 * i + 1];
    }
}

__device__ __forceinline__ int pad(const int i)
{
    return i + i / 16;
}

__device__ void do_scan_padding(float *g_odata, float *g_idata, const int n, float *data)
{
    int base = 2 * blockDim.x * blockIdx.x;
    int i = threadIdx.x;
    if (2 * i < n)
    {
        data[pad(i)] = g_idata[base + i];
    }
    int k = i + (n + 1) / 2;
    if (k < n)
    {
        data[pad(k)] = g_idata[base + k];
    }
    __syncthreads();
    int depth_power;
    for (depth_power = 1; depth_power < n; depth_power <<= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset < n)
        {
            data[pad(offset)] += data[pad(offset - depth_power)];
        }
        __syncthreads();
    }
    depth_power >>= 1;
    for (; depth_power >= 1; depth_power >>= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset + depth_power < n)
        {
            data[pad(offset + depth_power)] += data[pad(offset)];
        }
        __syncthreads();
    }
    if (2 * i < n)
    {
        g_odata[base + i] = data[pad(i)];
    }
    if (k < n)
    {
        g_odata[base + k] = data[pad(k)];
    }
}

__global__ void scan_padding(float *g_odata, float *g_idata, const int n)
{
    assert(blockIdx.x == 0);
    extern __shared__ float data[];
    do_scan_padding(g_odata, g_idata, n, data);
}

__global__ void scan_padding_sums(float *g_odata, float *g_idata, const int n, float *g_sums)
{
    extern __shared__ float data[];
    const int block = 2 * blockDim.x;
    const int blockN = min(block, n - block * blockIdx.x);
    do_scan_padding(g_odata, g_idata, blockN, data);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        g_sums[blockIdx.x] = g_odata[block * blockIdx.x + blockN - 1];
    }
}

__global__ void add_sums(float *g_data, float *g_sums, const int n)
{
    const int block = 2 * blockDim.x;
    const int base = (blockIdx.x + 1) * block;
    const int blockN = min(block, n - base);
    int i = threadIdx.x;
    if (2 * i < blockN)
    {
        g_data[base + i] += g_sums[blockIdx.x];
    }
    int k = i + (blockN + 1) / 2;
    if (k < blockN)
    {
        g_data[base + k] += g_sums[blockIdx.x];
    }
}
