print(''' ----------------------------------------------------
NK search (Version 1.0)

----------------------------------------------------

By Ron Tidhar & Tim Ott

----------------------------------------------------''')

'''
Given a set of Gavetti and Levinthal-type NK landscapes (see NK_Model_GandL.py), 
this script creates a library of analogies using local search
'''

# *** IMPORTS ***

import numpy as np
import itertools
from time import time
import os
import sys

def main():    
    try:
        NK = np.load(sys.argv[1])
    except IndexError:
        print("ERROR: Please input a valie NK landscape (.npy file)")
        print("You can use NK_Model_basic.py to generate a set of landscapes")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Please input a valie NK landscape (.npy file)")
        print("You can use NK_Model_basic.py to generate a set of landscapes")
        sys.exit(1)

    '''
    P = number of policies
    D = number of decisions
    X = number of observed variables
    N = P*D + X
    NK has dimensions (i, 2**N, N*2+2), where
      i     = number of landscapes
      2**N  = the number of possible decision sets
      N*2+2 = the value of each point on each landscape.
                - the first N columns are for the combinations of N decision variables (DV)
                    (hence we have 2**N rows for the total number of possible combinations)
                - the second N columns are for the contribution values of each DV
                - the next valuer is for the total fit (avg of N contributions)
                - the last one is to find out whether it is the local peak (0 or 1)
    '''
    P = int(input("Please provide the number of policies: "))
    D = int(input("Please provide the number of decisions per policy: "))
    N = int(np.log2(NK.shape[1]))
    X = int(N - P*D)
    i = NK.shape[0]
    Power_key = powerkey(N)
    policy_fitness = {}

    print(P,D,N,X)
    print(NK.shape)

    for obs_var in itertools.product(range(2), repeat=X):
        NK_obs = NK[:, range(np.sum(obs_var*Power_key[:X]), 2**N, 2**X),:]
        print(NK_obs.shape)

        for land in range(i):
            (policy, fit) = local_search_policy(N, P, D, NK_obs[land], Power_key)

            if policy in policy_fitness.items():
                policy_fitness[policy][fitness] += fit
                policy_fitness[policy][count] += 1
            else:
                policy_fitness[policy][fitness] = fit
                policy_fitness[policy][count] = 1

# FUNCTIONS

def powerkey(N):
    '''
    Used to find the location on the landscape for a given combination
    of the decision variables. Maps a decision string to the correct row in the NK_land matrix.
    Returns a vector with the powers of two: (2^(N-1), ..., 1)
    '''
    Power_key = np.power(2, np.arange(N - 1, -1, -1))
    return(Power_key)

def calc_fit(N, NK_land, inter_m, Current_position, Power_key):
    '''
    Takes landscape and a combination and returns a vector of fitness
    values (contribution value for each of the N decision variables)
    '''
    Fit_vector = np.zeros(N)
    for ad1 in np.arange(N):
        Fit_vector[ad1] = NK_land[np.sum(Current_position * inter_m[ad1] * Power_key), ad1]
    return(Fit_vector)

def local_search_policy(N, P, D, NK, Power_key):
    dec = random_start(P*D)
    
    curr_fit = NK[int(np.sum(dec*Power_key[:P*D])), 2*P*D]
    new_dec = dec.copy()
    new_dec[0] = abs(dec[0] - 1)
    count = 0

    while(True):
        count += 1
        new_dec = local_step(N, NK, dec, Power_key)
        if (all(new_dec == dec)):
            dec = new_dec
            break
        else:
            dec = new_dec
        policy = return_policy(P, D, decision)
    return policy, NK[np.sum(dec*Power_key), 2*N]

def local_step(N, NK, Current_position, Power_key):
    ''' 
    Local search strategy operates by changing only one decision 
    to reach a better position in the solution space.
    Thus, for a given NK instance and decision set, 
    we randomly iterate through neighbours, and go to the first, better option
    (In keeping with Levinthal, 1997, we don't assume the agent goes to the 
    highest-valued neighbour)
    '''

    # first make sure we're not at a local peak (if yes, we're done)
    if not NK[np.sum(Current_position*Power_key), 2*N+1]:
        Indexes = np.arange(N)
        np.random.shuffle(Indexes)

        Current_fit = NK[np.sum(Current_position*Power_key), 2*N]
        New_position = Current_position.copy()

        for new_dec in Indexes:  # check for the neighbourhood
            New_position[new_dec] = abs(New_position[new_dec] - 1)
            
            if (NK[np.sum(New_position*Power_key), 2*N] > Current_fit):
                # We have found a better position          
                return New_position
            # We didn't find a better position => change decision back
            New_position[new_dec] = abs(New_position[new_dec] - 1)

    # If we're here, we must be at a local optimum
    return Current_position

def return_policy(P, D, decision):
    policy = []
    for pp in P:
        pol = 0
        for dec in D:
            pol += decision[pp*D + dec]
        
        policy.append(int(pol > int(D/2)))

def random_start(N):
    return np.random.randint(2, size=N)

if __name__ == '__main__':
    main()
