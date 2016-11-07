print(''' ----------------------------------------------------
NK model (Version 3.2)

----------------------------------------------------

By Ron Tidhar & Tim Ott - adapted from Maciej Workiewicz (2014)

----------------------------------------------------''')

'''
This script creates i (i=100) NK landscapes (with N and K chosen by the user)
The NK landscapes are saved as a binary file (.npy)
'''

# *** IMPORTS ***

import numpy as np
import itertools
from time import time
import os

def main():    
    # *** MODEL INPUTS ***

    # SYSTEM INPUT
    i = 100  # number of landscapes to produce (use 100 or less for testing)

    # USER INPUTS

    cwd = os.getcwd()
    cwd += "/"
    print("Saving outputs to ", cwd)

    P = int(input("Enter a value for P, the number of policies: "))
    D = int(input("Enter a value for D, the number of decisions within a policy: "))
    X = int(input("Enter a value for X, the number of observed variables: "))
    
    X_prob = np.zeros(X)

    for xx in range(X):
        X_prob[xx] = float(input("Enter a probability for the observed variable's importance: "))

    N = int(P*D + X)

    K_within = float(input("Input K_within (decimal from 0 to 1): "))
    K_between = float(input("Input K_between (decimal from 0 to 1): "))
    
    Int_matrix = matrix_rand(P, D, K_within, K_between, X_prob)

    # *** GENERAL VARIABLES AND OBJECTS ***

    # For each iteration i

    Power_key = powerkey(N)
    NK = np.zeros((i, 2**N, N*2+2))
    print(i)
    for i_1 in np.arange(i):
        '''
        First create the landscapes for the DDs and PDs
        '''
        NK_land = nkland(N)
        NK[i_1] = comb_and_values(N, P, D, NK_land, Power_key, Int_matrix)    

    filename = 'NK_landscape_P_' + str(P) + '_D_' + str(D) + '_K_within_' + str(K_within) +
            '_K_between_' + str(K_between) + '_i_' + str(i) + '.npy'
    np.save(cwd + filename, NK)
    print("Saved landscapes as file " + filename)
    '''
    This saves the landscape into a numpy binary file
    '''

# FUNCTIONS AND INTERACTION MATRIX

def matrix_rand(P, D, K_within, K_between, X_prob):
    '''
    This function takes the number of policies, decisions and interdependencies 
    within policies (according to the probability K_within) and 
    across policies (according to the probability K_between). 
    It then creates a random interaction matrix with a diagonal filled.
    '''

    Int_matrix_rand = np.zeros((int(P*D), int(P*D) + len(X_prob))) # The last X columns are for observed variables
    for i in range(int(P*D)):
        for j in range(int(P*D) + len(X_prob)):
            if j < int(P*D):
                if (i == j):
                    Int_matrix_rand[i,j] = 1
                elif (int(i/D) == int(j/D)): # within same policy
                    Int_matrix_rand[i,j] = int(np.random.rand(1) < K_within)
                else:
                    Int_matrix_rand[i,j] = int(np.random.rand(1) < K_between)
            else:
                Int_matrix_rand[i,j] = int(np.random.rand(1) < X_prob[j-int(P*D)])
    return(Int_matrix_rand)

#ADDITIONAL FUNCTIONS

def powerkey(N):
    '''
    Used to find the location on the landscape for a given combination
    of the decision variables. Maps a decision string to the correct row in the NK_land matrix.
    Returns a vector with the powers of two: (2^(N-1), ..., 1)
    '''
    Power_key = np.power(2, np.arange(N - 1, -1, -1))
    return(Power_key)


def nkland(N):
    '''
    Generates an NK landscape - an array of random numbers ~U(0, 1).
    '''
    NK_land = np.random.rand(2**N, N)
    return(NK_land)

def comb_and_values(N, P, D, NK_land, Power_key, inter_m):
    '''
    Calculates values for *all combinations* on the landscape.
    - the first N columns are for the combinations of N decision variables (DV)
        (hence we have 2**N rows for the total number of possible combinations)
    - the second N columns are for the contribution values of each DV
    - the next valuer is for the total fit (avg of N contributions)
    - the last one is to find out whether it is the local peak (0 or 1)
    '''
    Comb_and_value = np.zeros((2**N, 2*N+2))
    c1 = 0  # starting counter for location

    # iterator creates bit strings for all possible combinations (i.e. positions)
    for c2 in itertools.product(range(2), repeat=N):
        '''
        this takes time so careful
        '''
        Combination1 = np.array(c2)  # taking each combination (convert iterator to an array)
        fit_1 = calc_fit(N, P, D, NK_land, inter_m, Combination1, Power_key)
        Comb_and_value[c1, :N] = Combination1  # combination and values
        Comb_and_value[c1, N:2*N] = fit_1
        Comb_and_value[c1, 2*N] = np.mean(fit_1)
        c1 = c1 + 1

    # now let's see if it's a local peak
    for c3 in np.arange(2**N):  
        Comb_and_value[c3, 2*N+1] = 1  # assume it is
        for c4 in np.arange(N):  # check for the neighbourhood
            new_comb = Comb_and_value[c3, :N].copy()
            new_comb[c4] = abs(new_comb[c4] - 1)
            if ((Comb_and_value[c3, 2*N] < Comb_and_value[int(np.sum(new_comb*Power_key)), 2*N])):
                Comb_and_value[c3, 2*N+1] = 0  # if smaller than the neighbour then not peak
                break
    return(Comb_and_value)

def calc_fit(N, P, D, NK_land, inter_m, Current_position, Power_key):
    '''
    Takes landscape and a combination and returns a vector of fitness
    values (contribution value for each of the N decision variables)
    '''
    Fit_vector = np.zeros(N)

    for ad1 in np.arange(N):
        if ad1 < int(P*D):
            Fit_vector[ad1] = NK_land[int(np.sum(Current_position * inter_m[ad1] * Power_key)), ad1]
        else:
            Fit_vector[ad1] = NK_land[int(np.sum(Current_position * Power_key)), ad1]
    return(Fit_vector)

def local_search(N, NK, Current_position, Power_key):
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
