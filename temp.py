import numpy as np
from scipy.stats import entropy

def fun1(x, c):
    value_counts = np.unique(x, return_counts=True)[1]
    assert c >= len(value_counts)
    return len(x) * np.log(c)

def fun2(x, c):
    value_counts = np.unique(x, return_counts=True)[1]
    assert c >= len(value_counts)
    assert value_counts.sum() == len(x)
    prob = value_counts / len(x)
    return entropy(prob, base=None)

def fun3(x, c):
    value_counts = np.unique(x, return_counts=True)[1]
    assert c >= len(value_counts)
    assert value_counts.sum() == len(x)
    prob = value_counts / len(x)
    log_prob = np.log(prob + (prob == 0)) 
    return -np.sum(prob * log_prob)

def fun4(x, c):
    value_counts = np.unique(x, return_counts=True)[1]
    if c > len(value_counts):
        value_counts = np.append(value_counts, [0] * (c - len(value_counts)))
    assert value_counts.sum() == len(x)
    prob = value_counts / len(x)
    log_prob = np.log(prob + (prob == 0)) 
    return -np.sum(prob * log_prob)

def fun5(x, c, y= np.array([390, 0.043, 13.25, 0.043, 0, 390, 0.043, 0, 119911.191, 0.043, 0])):
    hist, _, _ = np.histogram2d(x, y)
    prob = hist.sum(axis=1)/np.sum(hist)
    log_prob = np.log(prob + (prob == 0)) 
    return -np.sum(prob * log_prob)

x = np.array([0, 1, 3, 1, 2, 0, 1, 2, 4, 1, 2])
x = np.array([0, 1, 3, 1, 2, 0, 1, 2, 4])

print("fun1(x, 5):", fun1(x, 5))
print("fun1(x, 6):", fun1(x, 5))
print("fun1(x, 7):", fun1(x, 5))
print("fun1(x, 8):", fun1(x, 5))
print("fun2(x, 5):", fun2(x, 5))
print("fun2(x, 6):", fun2(x, 5))
print("fun2(x, 7):", fun2(x, 5))
print("fun2(x, 8):", fun2(x, 5))
print("fun3(x, 5):", fun3(x, 5))
print("fun3(x, 6):", fun3(x, 5))
print("fun3(x, 7):", fun3(x, 5))
print("fun3(x, 8):", fun3(x, 5))
print("fun4(x, 5):", fun4(x, 5))
print("fun4(x, 6):", fun4(x, 5))
print("fun4(x, 7):", fun4(x, 5))
print("fun4(x, 8):", fun4(x, 5))
print("fun5(x, 5):", fun5(x, 5))
print("fun5(x, 6):", fun5(x, 5))
print("fun5(x, 7):", fun5(x, 5))
print("fun5(x, 8):", fun5(x, 5))

for i in range(10000):
    length = np.random.randint(2, 1000)
    np.random.seed(i)
    x_bottom, x_top = tuple(np.sort(np.random.choice(range(-100, 100), 2, replace=False)))
    print("[", i, "]", "len:", length, "| x: [", x_bottom, x_top, " ]")
    possible_values = x_top - x_bottom
    x = np.random.randint(low=x_bottom, high=x_top, size=length)
    if np.isclose(fun2(x, possible_values), fun3(x, possible_values)):
        if np.isclose(fun2(x, possible_values), fun4(x, possible_values)):
            if np.isclose(fun2(x, possible_values), fun5(x, possible_values)):
                if fun2(x, possible_values) <= fun1(x, possible_values):
                    print("OK!")
                else:
                    print("fun2 > fun1:", x)
            else:
                print("fun2 != fun5:", x)
        else:
            print("fun2 != fun4:", x)
    else:
        print("fun2 != fun3:", x)