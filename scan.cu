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

__device__ __forceinline__ void add_constant(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__device__ __forceinline__ void add_scans4(float4 &to, float4 &from)
{
    float sum = from.w;
    to.x += sum;
    to.y += sum;
    to.z += sum;
    to.w += sum;
}

__device__ __forceinline__ void scan4_into(float4 &to, float4 &from)
{
    to.x = from.x;
    to.y = to.x + from.y;
    to.z = to.y + from.z;
    to.w = to.z + from.w;
}

__device__ void do_scan_padding4(float4 *g_odata, float4 *g_idata, const int n4, float4 *data)
{
    int base = 2 * blockDim.x * blockIdx.x;
    int i = threadIdx.x;
    if (2 * i < n4)
    {
        scan4_into(data[pad(i)], g_idata[base + i]);
    }
    int k = i + (n4 + 1) / 2;
    if (k < n4)
    {
        scan4_into(data[pad(k)], g_idata[base + k]);
    }
    __syncthreads();
    int depth_power;
    for (depth_power = 1; depth_power < n4; depth_power <<= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset < n4)
        {
            add_scans4(data[pad(offset)], data[pad(offset - depth_power)]);
        }
        __syncthreads();
    }
    depth_power >>= 1;
    for (; depth_power >= 1; depth_power >>= 1)
    {
        int offset = 2 * depth_power * (i + 1) - 1;
        if (offset + depth_power < n4)
        {
            add_scans4(data[pad(offset + depth_power)], data[pad(offset)]);
        }
        __syncthreads();
    }
    if (2 * i < n4)
    {
        g_odata[base + i] = data[pad(i)];
    }
    if (k < n4)
    {
        g_odata[base + k] = data[pad(k)];
    }
}

__global__ void scan_padding4(float4 *g_odata, float4 *g_idata, const int n4)
{
    assert(blockIdx.x == 0);
    extern __shared__ float4 data4[];
    do_scan_padding4(g_odata, g_idata, n4, data4);
}

__global__ void scan_padding_sums4(float4 *g_odata, float4 *g_idata, const int n4, float *g_sums)
{
    extern __shared__ float4 data4[];
    const int block = 2 * blockDim.x;
    const int blockN4 = min(block, n4 - block * blockIdx.x);
    do_scan_padding4(g_odata, g_idata, blockN4, data4);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        g_sums[blockIdx.x] = g_odata[block * blockIdx.x + blockN4 - 1].w;
    }
}

__global__ void add_sums4(float4 *g_data, float *g_sums, const int n4)
{
    const int block = 2 * blockDim.x;
    const int base = (blockIdx.x + 1) * block;
    const int blockN4 = min(block, n4 - base);
    float sum = g_sums[blockIdx.x];
    int i = threadIdx.x;
    if (2 * i < blockN4)
    {
        add_constant(g_data[base + i], sum);
    }
    int k = i + (blockN4 + 1) / 2;
    if (k < blockN4)
    {
        add_constant(g_data[base + k], sum);
    }
}
