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

__global__ void scan(float *g_odata, float *g_idata, const int n)
{
    extern __shared__ float data[];
    int i = threadIdx.x;
    if (2 * i < n)
    {
        data[2 * i] = g_idata[2 * i];
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
        if (offset < n)
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

__global__ void scan_padding(float *g_odata, float *g_idata, int n)
{
    extern __shared__ float data[];
    int i = threadIdx.x;
    if (2 * i < n)
    {
        data[pad(2 * i)] = g_idata[pad(2 * i)];
        data[pad(2 * i + 1)] = g_idata[pad(2 * i + 1)];
    }
    __syncthreads();
    int depth_power = 1;
    for (; depth_power < n; depth_power <<= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset < n)
        {
            data[pad(offset)] += data[pad(offset - depth_power)];
        }
        __syncthreads();
    }
    for (depth_power >>= 1; depth_power >= 1; depth_power >>= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset < n)
        {
            data[pad(offset + depth_power)] += data[pad(offset)];
        }
        __syncthreads();
    }
    if (2 * i < n)
    {
        g_odata[2 * i] = data[pad(2 * i)];
        g_odata[2 * i + 1] = data[pad(2 * i + 1)];
    }
}