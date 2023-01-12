nums = [3, 1, 7, 0, 4, 1, 6, 3]
def cumsum(x):
    result = []
    s = 0
    for a in x:
        s += a
        result.append(s)
    return result
assert cumsum([3, 2, 1]) == [3, 5, 6]
assert cumsum([3, 1, 7]) == [3, 4, 11]

T = 16
def scan_slow(x, n):
    assert len(x) == n
    nums = [list(x), [None] * n]
    d_power = 1
    # double buffering
    a = 0
    b = 1
    while d_power < n:
        for i in range(T): # gpu simulator
            if i < n:
                if i >= d_power:
                    nums[b][i] = nums[a][i] + nums[a][i - d_power]
                else:
                    nums[b][i] = nums[a][i]
        d_power *= 2
        a, b = b, a
    return nums[a]
assert scan_slow(nums, len(nums)) == cumsum(nums)

f = lambda x, d_power: 2*d_power*(x + 1) - 1
ff = lambda d_power: [f(x, d_power) for x in range(4)]
assert ff(1) == [1, 3, 5, 7], ff(1)
assert ff(2) == [3, 7, 11, 15], ff(2)
f = lambda x, d_power: 2*d_power*(x + 1) - 1 - d_power
assert ff(1) == [0, 2, 4, 6], ff(1)
assert ff(2) == [1, 5, 9, 13], ff(2)

def scan(x, n):
    nums = list(x)
    d_power = 1
    while d_power < n:
        for i in range(T): # GPU similator
            if d_power * i < n // 2:
                nums[2 * d_power * (i + 1) - 1] += nums[2 * d_power * (i + 1) - 1 - d_power]
        d_power *= 2

    d_power //= 2
    while d_power >= 1:
        for i in range(T): # GPU similator
            if d_power <= d_power * (i + 1) < n // 2:
                nums[2 * d_power * (i + 1) - 1 + d_power] += nums[2 * d_power * (i + 1) - 1]
        d_power //= 2
    return nums
assert scan(nums, len(nums)) == cumsum(nums)

BANKS = 4
def scan_padding(x, n):
    nums = [-1] * (T * BANKS)
    
    for i in range(T): # GPU similator
        if i < n:
            nums[i + i // BANKS] = x[i]

    d_power = 1
    while d_power < n:
        for i in range(T): # GPU similator
            if d_power * i < n // 2:
                ai = 2 * d_power * (i + 1) - 1
                bi = 2 * d_power * (i + 1) - 1 - d_power
                nums[ai + ai // BANKS] += nums[bi + bi // BANKS]
        d_power *= 2

    d_power //= 2
    while d_power >= 1:
        for i in range(T): # GPU similator
            if d_power <= d_power * (i + 1) < n // 2:
                ai = 2 * d_power * (i + 1) - 1 + d_power
                bi = 2 * d_power * (i + 1) - 1
                nums[ai + ai // BANKS] += nums[bi + bi // BANKS]
        d_power //= 2

    out = [-1] * n
    for i in range(T): # GPU similator
        if i < n:
            out[i] = nums[i + i // BANKS]

    return out
assert scan_padding(nums, len(nums)) == cumsum(nums)
