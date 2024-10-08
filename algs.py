def opt_s(arr, g, b):
    s = np.sum(arr)
    if g*b >= 1:
        return s/g
    else:
        return s + s*b*(1-g)
    
def nf_s(arr, g, b):
    s = np.sum(arr)
    threshold = min(g + (1/b), 1)
    return (3*s)/threshold

def gaaf_s(arr, g, b):
    if g + 1/b >= 1:
        return 1.7 * opt_s(arr, g, b)
    elif g*b >= 1:
        return 2*np.sum(arr) / (g+ (1/b))
    else:
        return opt_s(arr, g, b)
    

def green_next_fit(arr, g, b):
    curr = 0
    cost = 1
    threshold = min(g + (1/b), 1)
    for item in arr:
        if curr + item > threshold:
            cost += max((curr - g), 0)*b + 1
            curr = item
        else:
            curr += item
    
    return cost

def green_worst_fit(arr,g,b):
    cost = 1
    bins = [0]
    threshold = min(g + (1/b), 1)
    for item in arr:
        mini = min(bins)
        if mini + item <= threshold:
            bins[bins.index(mini)] += item
        else:
            cost += 1
            bins.append(item)
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def green_best_fit(arr,g,b):
    cost = 1
    bins = [0]
    threshold = min(g + (1/b), 1)
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(max(allowed))] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def green_first_fit(arr,g,b):
    cost = 1
    bins = [0]
    threshold = min(g + (1/b), 1)
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(allowed[0])] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def best_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]
    if g*b <= 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(max(allowed))] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost
    
    return cost

def first_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]
    if g*b <= 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        allowed = list(filter(lambda n: n <= (threshold - item), bins))
        if len(allowed) == 0:
            cost += 1
            bins.append(item)
        else:
            bins[bins.index(allowed[0])] += item
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost

def next_fit(arr, g, b, threshold=1):
    curr = 1
    cost = 0
    if g*b <= 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        if curr + item > threshold:
            cost += max((curr - g), 0)*b + 1
            curr = item
        else:
            curr += item
    
    return cost


def gamma(i, add_one_denom=False):

    if i == -1:
        if add_one_denom:
            return 1
        return 1.691
    if i == 0:
        if add_one_denom:
            return 0.5
        return 1
    
    if add_one_denom:
        return gamma(i) - gamma(i+1)
    
    else:
        return  gamma(i-1)**2 / (gamma(i-1) + 1)
    

def sum_gamma(stop, add_one_denom=False):
    sum = 0

    for i in range(stop):
        sum += gamma(i, add_one_denom)

    return sum

def Har_CR(g, b):

    if g == 1:
        return 1.691
    else:
        i_star = min(5,math.floor(g/(1-g)))

    numerator = b * (1 - sum_gamma(i_star, add_one_denom=True) - g*(1.691 - sum_gamma(i_star))) + 1.691
    if abs(g - 0.5) < 0.1:
        print("black part of numerator:",(numerator - 1.691)/b)
        print("value of g:", g)
    denom = b*(1-g) + 1

    return numerator/denom 

def combined_cr_UB(g,b):
    if g*b < 1:
        return Har_CR(g,b)
    if g*b > 17/3:
        return 1.7
    if g + 1/b > 1:
        return 1.7 * g * (3/17 * (1-g)*b + 1)
    else:
        return 2*g/(g + 1/b)


def Har(arr, g, b, max_i=100):
    if g == 1:
        i_star = max_i
    else:
        i_star = min(math.floor(g/(1-g)), max_i)
    cost = 0

    print(i_star)
    print(max_i)
    print(i_star + max_i)

    bins = np.zeros((max_i + i_star, 3)) # Each bin is a tuple (weight, count, i)
    for i in range(1, max_i+1):
        if i <= i_star:
            bins[2*i - 1][2] = i
            bins[2*i - 2][2] = i
        else:
            bins[i + i_star - 1][2] = i

    for item in arr:
        item_index = -1
        for i in range(1, max_i+1):
            if i <= i_star:
                if (1/(i+1)) <= item and item < (g/i):
                    item_index = 2*i - 1
                elif (g/i) <= item and item < (1/i):
                    item_index = 2*i - 2
            elif (1/(i+1)) <= item and item < (1/i):
                item_index = i + i_star - 1
        
        new_bin, item_cost = add_item_har(bins[item_index], item, g, b)
        bins[item_index] = new_bin
        cost += item_cost

    return cost, bins


def add_item_har(bin, item, g, b):
    cost = 0
    if bin[0] == 0:
        bin[0] = item
        bin[1] = 1
        cost += max(item - g, 0) * b + 1
        return bin, cost
    
    if bin[1] == bin[2]:
        cost += 1
        bin[1] = 1
        bin[0] = item

    else:
        cost += max(item - max(g - bin[0], 0), 0) * b
        bin[0] += item
        bin[1] += 1

    return bin, cost

def worst_fit(arr,g,b, threshold=1):
    cost = 1
    bins = [0]

    if g*b <= 1:
        threshold = 1
    else:
        threshold = g
    for item in arr:
        mini = min(bins)
        if mini + item <= threshold:
            bins[bins.index(mini)] += item
        else:
            cost += 1
            bins.append(item)
            
    for bin in bins:
        cost += max(bin - g, 0) * b
    
    return cost
