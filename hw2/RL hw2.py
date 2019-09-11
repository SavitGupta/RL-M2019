#!/usr/bin/env python
# coding: utf-8

# In[235]:


import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import math
import copy, random
inf = math.inf


# In[242]:


def round1(x):
    return round(x, 1)
# Evaluates policy by solving linear equations
def policy_eval_linear(pi, discount, result):
    n = len(pi)
    A = np.zeros([n, n]) #co-ffciant Matrix
    C = np.zeros(n) #constant matrix
    actions = pi[0].keys()
    #iterating over all states
    for cur in range(n):
        A[cur, cur] -= 1
        c = 0
        for a in actions:
            probs = result(cur, a) #returns a list of possibilities with thier probabilities
            for prob, r, s in probs:
                c -= prob * r * pi[cur][a]
                A[cur, s] += prob * pi[cur][a] * discount 
        C[cur] = c        
    V = np.linalg.solve(A, C)    
    eps = 1e-10
    return V

#Utility Function to print a single, array into a 2-d array, with col-size = n
def print_grid(arr, n):
    m = n 
    for i in range(n):
        for j in range(m):
            print(arr[i*m + j], end = ', ')
        print()


def policy_improv(pi, v, discount, result):
    n = len(pi) #number of states
    actions = pi[0].keys()
    default = {} #utility data member: used to initialize the policy of each action with all zeros.
    states = pi.keys()
    for a in actions:
        default[a] = 0
    for i in states:
        best = [-inf, '0']
        for a in actions:
            probs = result(i, a) #returns a list of possiblities with >0 probability
            qi = 0
            for prob, r, state in probs:
                qi += prob*(r + discount*v[state])
            best = max(best, [qi, a])
        pi[i] = default.copy()
        pi[i][best[1]] = 1    
    return pi


def policy_eval(pi, discount, result):
    n = len(pi)
    actions = pi[0].keys()
    v = np.zeros(n)
    old_v = np.zeros(n)
    states = pi.keys()
    delta = 1e-3
    diff = 1000
    while(diff > delta):
        diff = 0
        old_v = copy.deepcopy(v) 
        for i in states:
            vi  = 0
            for a in actions:
                probs = result(i, a)
                for prob, r, state in probs:
                    vi += pi[i][a]*prob*(r + discount*old_v[state])
              
            v[i] = vi
            diff = max(diff, abs(v[i] - old_v[i]))
    return v
                
    
def policy_iter(pi, discount, result, lin = True):
    log = []
    old_v = np.array([])
    flag = True
    eps = 1e-10
    
    #Choice between solving using linear equations and policy method
    if(lin):
        v = policy_eval_linear(pi, discount, result)
    else:
        
        v = policy_eval(pi, discount, result)
    pi = policy_improv(pi, v, discount, result)
    
    
    while(flag or (old_v != v).any()):
        old_v = v
        flag = False
        if(lin):
            v = policy_eval_linear(pi, discount, result)
        else:
            v = policy_eval(pi, discount, result)
        pi = policy_improv(pi, v, discount, result)
        v = np.array(list(map(round1, v)))
        opt = {}
        for i in pi:
            for j in pi[i]:
                if(pi[i][j] > 1 - eps):
                    opt[i] = j
        log.append((copy.deepcopy(v), copy.deepcopy(opt)))
    return pi, log
        
def value_iter(pi, discount, result):
    log = []
    delta = 1e-3
    diff = 1000
    actions = pi[0].keys()
    n = len(pi)
    old_v = []
    v = np.zeros(n)
    default = {}
    states = pi.keys()
    opt = {}

    for a in actions:
        default[a] = 0
    while(diff > delta):
        diff = 0
        old_v = v.copy()
        for i in states:
            best = [-math.inf,'0']
            for a in actions:
                probs = result(i, a)
                qi = 0
                for prob, r, state in probs:
                    qi += prob * (r + discount*old_v[state])
                best = max(best, [qi, a])
            v[i] = best[0]
            opt[i] = best[1]
            
            diff = max(diff, abs(old_v[i] - v[i]))
        log.append((copy.deepcopy(v), copy.deepcopy(opt)))
    for i in states:
        best = [-inf, '0']
        for a in actions:            
            probs = result(i, a)
            qi = 0
            for prob, r, s in probs:
                
                qi += prob * (r + discount*v[s])
            best = max(best, [qi, a])
        pi[i] = default.copy()
        pi[i][best[1]] = 1
    return pi, log
    
            

    
   


# # Q2, 4
# ##Driver and Utility of function
# ##Also Has the MDP in the form of result function

# In[243]:


def result(state, a):
    n = 5
    srca, desta, srcb, destb = 1, 21, 3, 13
    if(state == srca):
        return [(1, 10, desta)]
    if(state == srcb):
        return [(1, 5, destb)]
    if(a == 'l'):
        return left(state, n)
    elif(a == 'r'):
        return right(state, n)
    elif(a == 'u'):
        return up(state, n)
    else:
        return down(state, n)
    
def up(state, n): #returns (r, s)
    i = state // n
    j  = state % n
    if(i > 0): 
        return [(1,0, state - n)] 
    else:
        return [(1,-1, state)]
    
def left(state, n): #returns (r, s)
    i = state // n
    j  = state % n
    if(j > 0): 
        return [(1,0, state - 1)]
    else:
        return [(1,-1, state)]
    
def down(state, n): #returns (r, s)
    i = state // n
    j  = state % n
    if(i < n - 1): 
        return [(1,0, state + n)] 
    else:
        return [(1,-1, state)]
    
def right(state, n): #returns (r, s)
    i = state // n
    j  = state % n
    if(j < n - 1): 
        return [(1,0, state + 1)] 
    else:
        return [(1,-1, state)]
def round1(x):
    return round(x, 1)

def q2():
    n = 5
    deci = {"l":0.25, "r":0.25, "u":0.25, "d":0.25}
    pi = []
    for i in range(n*n):
        pi.append(deci)
    v = policy_eval_linear(pi, 0.9, result)
    v = list(map(round1, v))
    print("value Function as solved by Linear Equations")
    print_grid(v, n)
    
def q4():
    n = 5
    deci = {"l":0.25, "r":0.25, "u":0.25, "d":0.25}
    pi = {}
    for i in range(n*n):
        pi[i] = copy.deepcopy(deci)
    
    pi_star, log = policy_iter(pi, 0.9, result)
    to_print = []
    #Simplifing the entire policy to just the best action for clearer display
    for i in pi_star:
        i = pi_star[i]
        best = (-1, -1)
        for j in deci.keys():
            best = max(best, (i[j], j))
        to_print.append(best[1])
    print("Optimal Policy")
    print_grid(to_print, n)
    v = policy_eval_linear(pi, 0.9, result)
    v = list(map(round1, v))
    print("value Function")
    print_grid(v, n)
q2()
q4()        
    
    
    


# # Q6
# ##Driver and Utility of function
# ##Also Has the MDP in the form of result function
# ##Bug is fixed by changing the check of the equality, since it is possible for 2 optimum policies to exisit, but the optimum value function is uniuqe.
# 

# In[244]:


def result_helper_q6(state, a):
    n = 4
    if(a == 'l'):
        return left(state, n)
    elif(a == 'r'):
        return right(state, n)
    elif(a == 'u'):
        return up(state, n)
    else:
        return down(state, n)
def print_log(log):
    print("some samples from log")
    if(len(log) < 3):
        for i in log:
            print_grid(i[0],4)
            print_grid(i[1],4)
            print()
    else:
        index = []
        for i in range(3):
            index.append(random.randint(0, len(log) - 1))
        index.sort()
        for i in index:
            i = log[i]
            print_grid(i[0],4)
            print_grid(i[1],4)
            print()

def result_q6(state, a):
    if(state in [0, 15]):
        return [(1,0, 0)]
    p,waste, state = result_helper_q6(state, a)[0]
    if(state in [0, 15]):
        return [(1,-1, 0)]
    return [(1,-1, state)]

def q6():
    n = 4
    deci = {"l":0.25, "r":0.25, "u":0.25, "d":0.25}
    pi = {}
    for i in range(n*n):#creating initial policy
        pi[i] = copy.deepcopy(deci)
    pi_star, log = policy_iter(pi, 1, result_q6, False)
    to_print = []
    print_log(log)
    #Simplifing the entire policy to just the best action for clearer display
    for i in pi_star:
        i = pi_star[i]
        best = (-1, -1)
        for j in deci.keys():
            best = max(best, (i[j], j))
        to_print.append(best[1])
    print("An optimal Policy using Policy Iteration")
    print_grid(to_print, n)
    v = policy_eval(pi_star, 1, result_q6)
    v = list(map(round1, v))
    print("Corresponding value function")
    print_grid(v, n)
    pi = {}
    for i in range(n*n):#creating initial policy
        pi[i] = copy.deepcopy(deci)
    pi_star2, log2 = value_iter(pi,1, result_q6)
    print_log(log2)
    to_print = [] 
    #Simplifing the entire policy to just the best action for clearer display
    for i in pi_star2:
        i = pi_star2[i]
        best = (-1, -1)
        for j in deci.keys():
            best = max(best, (i[j], j))
        to_print.append(best[1])
    print("An optimal policy using Value Iteration")
    print_grid(to_print, n)

    v = policy_eval(pi_star2, 1, result_q6)
    v = list(map(round1, v))
    print("Corresponding Value function")
    print_grid(v, n)
    print()
    
q6()
    


# In[218]:





def poisson_prob(lam, n):
    return (pow(lam, n)/math.factorial(n))*math.exp(-lam)
    

def result(s,a):
    
    ans = {}
    for i in range(20):
        probi = poisson_prob(3,i)
        for j in range(20):
            probj = poisson_prob(4, j)
            r = min(s[0], i) + min(s[1], j)
            r *= 10
            
            
            curx, cury = max(s[0] - i, 0), max(s[1] - j, 0)
            for retx in range(20):
                proby = poisson_prob(3, retx)
                for rety in range(20):
                    proby = poisson_prob(2, rety)
                    finx = min(20, minx + retx)
                    finy = min(20, miny + rety)
                    if((finx, finy) not in ans):
                            ans[(finx, finy)] = [0, 0]
                    ans[(finx, finy)][0] += probi*probj*probx*proby
                    ans[(finx, finy)][1] += r * probi*probj*probx*proby
                
                    
                    
                    
                    
                    
    for i in range(20):
        probi = poisson_prob(3,i)
        for j in range(20):
            probj = poisson_prob(2, i)
            
            
            
    
                
                
            
    
    
    







# In[ ]:




