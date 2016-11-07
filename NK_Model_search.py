print(''' ----------------------------------------------------
NK search (Version 1.0)

----------------------------------------------------

By Ron Tidhar & Tim Ott

----------------------------------------------------''')

'''
Given a set of NK landscapes (see NK_Model_basic.py), 
this script applies both local search and Gavetti and Levinthal's (2005) search strategies
Outputting the results with errors presented as one standard deviation
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
    except IndexError and FileNotFoundError:
        print("ERROR: Please input a valie NK landscape (.npy file)")
        print("You can use NK_Model_basic.py to generate a set of landscapes")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Please input a valie NK landscape (.npy file)")
        print("You can use NK_Model_basic.py to generate a set of landscapes")
        sys.exit(1)

    '''
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
    i = NK.shape[0]
    N = int(np.log2(NK.shape[1]))
    Power_key = powerkey(N)

    dec = random_start(N)

    print("starting = " + str(dec))

    num_steps = []
    final_fitness = []
    run_times = []

    for land in range(i):    
        (steps, fit) = local_search(N, NK[land], dec, Power_key)

        num_steps.append(steps)
        final_fitness.append(fit)

    print("Local Search results for " + str(i) + " iterations:")
    print("     Average # of iterations  = %.2f +/- %.2f (one std.dev)" % (np.mean(num_steps), np.std(num_steps)))
    print("     Average fitness          = %.2f +/- %.2f (one std.dev)" % (np.mean(final_fitness), np.std(final_fitness)))

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

def local_search(N, NK, dec, Power_key):
    curr_fit = NK[int(np.sum(dec*Power_key)), 2*N]
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
    return count, NK[np.sum(dec*Power_key), 2*N]

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

def random_start(N):
    return np.random.randint(2, size=N)

if __name__ == '__main__':
    main()
