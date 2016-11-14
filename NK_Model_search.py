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
    except IndexError:
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

    runs = int(input("Input the number of runs per landscape: "))
    filename = str(sys.argv[1])
    P = int(filename[filename.find("P_")+2])
    D = int(filename[filename.find("D_")+2])

    i = NK.shape[0]
    N = int(np.log2(NK.shape[1]))
    Power_key = powerkey(N)

    num_local_steps = []
    final_local_fitness = []
    
    num_dw_steps = []
    final_dw_fitness = []

    for land in range(i):   
        for _ in range(runs):

            start = random_start(N)

            (local_steps, local_fit) = local_search(start, N, NK[land], Power_key)
            (dw_steps, dw_fit) = decision_weaving(start, N, P, D, NK[land], Power_key)

            num_local_steps.append(local_steps)
            final_local_fitness.append(local_fit)

            num_dw_steps.append(dw_steps)
            final_dw_fitness.append(dw_fit)


    print("Local Search results for " + str(i) + " landscapes:")
    print_fitness(final_local_fitness)
    print_num_iterations(num_local_steps)

    print("Decision Weaving results for " + str(i) + " landscapes:")
    print_fitness(final_dw_fitness)    
    print_num_iterations(num_dw_steps)
    

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

def random_start(N):
    return np.random.randint(2, size=N)

def local_search(dec, N, NK, Power_key):
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

        Current_fit = fitness(Current_position, N, NK, Power_key)
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

def decision_weaving(Current_position, N, P, D, NK, Power_key):
    new_dec = Current_position.copy()
    new_dec[0] = abs(Current_position[0] - 1)
    count = 0

    while(True):
        unvisited_policies = np.arange(P)
        np.random.shuffle(unvisited_policies)

        for policy in unvisited_policies:
            count += 1
            new_dec = search_domain(N, P, D, NK, Current_position, policy, Power_key)
            new_dec = stepping_stone(N, P, D, NK, new_dec, policy, Power_key)
                
        if NK[np.sum(new_dec*Power_key), 2*N+1]:
            Current_position = new_dec
            break
        else:
            Current_position = new_dec

    return count, NK[np.sum(Current_position*Power_key), 2*N]    

def search_domain(N, P, D, NK, Current_position, policy, Power_key):
    #Current_fit = NK[np.sum(Current_position*Power_key), 2*N]
    Max_position = Current_position.copy()
    New_position = Current_position.copy()

    for pp in itertools.product(range(2), repeat=D):
    # check for other decision sets within domain (policy can change)
        New_position[policy*D:(policy+1)*D] = pp
        if (fitness(New_position, N, NK, Power_key) > fitness(Max_position, N, NK, Power_key)):
            # We have found a better position          
            return New_position
            # Max_position = New_position
    return Max_position

def stepping_stone(N, P, D, NK, Current_position, policy, Power_key):
    Current_fit = fitness(Current_position, N, NK, Power_key)
    New_position = Current_position.copy()

    stepping_stones = np.arange(P*D)
    np.delete(stepping_stones, list(range(policy*D, (policy+1)*D)))
    np.random.shuffle(stepping_stones)
    dec = stepping_stones[0]

    New_position[dec] = abs(New_position[dec] - 1)
            
    if (fitness(New_position, N, NK, Power_key) > Current_fit):
        # We have found a better position          
        return New_position
    else:
        return Current_position

def current_policy(P, D, decision):
    policy = []
    for pp in range(P):
        pol = 0
        for dec in range(D):
            pol += decision[pp*D + dec]
        
        policy.append(int(int(pol) > int(D/2))) # Policy is defined by majority decisions

    return policy

def fitness(position, N, NK, Power_key):
    return NK[np.sum(position*Power_key), 2*N]

def print_num_iterations(steps, indent = "     "):
    print(indent + "Average # of iterations  = %.2f +/- %.2f (one std.dev)" % (np.mean(steps), np.std(steps)))

def print_fitness(fitness, indent = "     "):
    print(indent + "Average fitness          = %.2f +/- %.2f (one std.dev)" % (np.mean(fitness), np.std(fitness)))

if __name__ == '__main__':
    main()
