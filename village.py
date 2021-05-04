import numpy as np

from matplotlib import pyplot as plt
from random import sample
from scipy.special import comb


# Perform 1000 simulations for different configurations of m and n
est = []
cis = []
for n in range(5):
    est_tmp = []
    cis_tmp = []
    for m in range(4, 100):
        its = []
        n_runs = 1000
        for _ in range(n_runs):
            it = 0
            p = [1] + [0] * m
            q = [0] * (m + 1)
            while sum(p) != 0 and sum(q) != m + 1:
                it += 1
                for i in [x for x in range(m + 1) if p[x]]:
                    for j in sample([x for x in range(m + 1) if x != i], n):
                        if p[j] == 0 and q[j] == 0:
                            p[j] = 1
                    p[i] = 0
                    q[i] = 1
            its.append(it)
        est_tmp.append(np.mean(its))
        cis_tmp.append(1.96 * np.std(its) / np.sqrt(n_runs))
    est.append(est_tmp)
    cis.append(cis_tmp)


# Plot the results
fig, ax = plt.subplots()
for n in range(5):
    x = range(4, 100)
    y = est[n]
    y_1 = [y[i] - cis[n][i] for i in range(96)]
    y_2 = [y[i] + cis[n][i] for i in range(96)]
    ax.plot(x, y, label=f"n = {n}")
    ax.fill_between(x, y_1, y_2, alpha=0.5)   
plt.ylim(0, 14)
plt.xlabel("Number of villagers")
plt.ylabel("Average survival time")
plt.legend()
plt.savefig("img_1.png")


# Calculate the expected value for $n = 1$
est[1] = [1 + sum(i ** 2 / (m ** i) * np.math.factorial(m - 1) / \
          np.math.factorial(m - i) for i in range(m + 1)) for m in \
          range(4, 100)]
cis[1] = [0] * 96


# Plot the results
fig, ax = plt.subplots()
for n in range(5):
    x = range(4, 100)
    y = est[n]
    y_1 = [y[i] - cis[n][i] for i in range(96)]
    y_2 = [y[i] + cis[n][i] for i in range(96)]
    ax.plot(x, y, label=f"n = {n}")
    ax.fill_between(x, y_1, y_2, alpha=0.5)   
plt.ylim(0, 14)
plt.xlabel("Number of villagers")
plt.ylabel("Average survival time")
plt.legend()
plt.savefig("img_2.png")


# Calculate the probability of l infectuous people infecting p healthy people
def prob_trans(m, n, k, l, p):
    global trans_probs
    if (k, l, p) in trans_probs:
        return trans_probs[(k, l, p)]
    if p == 0:
        s = (np.math.factorial(k - 1) * np.math.factorial(m - n) / \
             np.math.factorial(k - 1 - n) / np.math.factorial(m)) ** l
    elif l == 0:
        s = 0
    else:
        s = 0
        for i in range(min(p, n) + 1):
            x = y = z = 1
            for j in range(n - i):
                x *= k - 1 - j
            for j in range(i):
                y *= m - k + 1 - j
            for j in range(n):
                z *= m - j
            s += x * y / z * comb(n, i) * prob_trans(m, n, k + i, l - 1, p - i)
    trans_probs[(k, l, p)] = s
    return s


# Calculate the probability of ending up in a given state
def prob_state(state):
    global state_probs
    if state in state_probs:
        return state_probs[state]
    global markov_probs
    if state not in markov_probs:
        return 0
    d = markov_probs[state]
    s = sum(d[prev] * prob_state(prev) for prev in d)
    state_probs[state] = s
    return s


# Calculate the expected survival time for different configurations of m and n
calc = []
for n in range(5):
    calc_tmp = []
    for m in range(4, 100):
        trans_probs = {}
        state_probs = {(1, n, 1): 1}
        markov_probs = {(1, n, 1): {(0, 1, 0): 1}}
        new_states = set([(1, n, 1)])
        while new_states:
            state = new_states.pop()
            t, l, w = state
            if l == 0 or w > m:
                continue
            for p in range(min(n * l, m - t) + 1):
                prob = prob_trans(m, n, t + l, l, p)
                new_state = (t + l, p, w + 1)
                if prob > 0:
                    new_states.add(new_state)
                if new_state not in markov_probs:
                    markov_probs[new_state] = {}
                markov_probs[new_state][state] = prob
        calc_tmp.append(sum(j * sum(prob_state((i, 0, j)) for i in \
                            range(m + 2)) for j in range(m + 2)))
        print(n, m, calc_tmp[-1])
    calc.append(calc_tmp)


# Plot the results
fig, ax = plt.subplots()
x = range(4, 100)
for n in range(5):
    y = calc[n]
    ax.plot(x, y, label=f"n = {n}")
plt.ylim(0, 14)
plt.xlabel("Number of villagers")
plt.ylabel("Average survival time")
plt.legend()
plt.savefig("img_3.png")


# Perform curve fitting for n = 1 and n = 2
print(np.polyfit(np.sqrt(x), calc[1], 1))
print(np.polyfit(np.log(x), calc[2], 1))
