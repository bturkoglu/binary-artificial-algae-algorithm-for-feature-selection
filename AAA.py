
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 2 2022

@authors: Bahaeddin Turkoglu, Sait Ali Uymaz and Ersin Kaya
"""

import random
import numpy
import math
from solution import solution
import time
import transfer_functions_benchmark
import fitnessFUNs

def AAA(objf, lb, ub, dim, SearchAgents_no, Max_iter,trainInput,trainOutput):

    iter=Max_iter
    Max_iter = Max_iter * SearchAgents_no

    s_force = 2             # share force paramater
    e_loss = 0.3            # energy loss paramater
    ap = 0.2                # adaptasyon paramater

    starveAlg=numpy.zeros(SearchAgents_no,'int')
    alg_size_matrix=numpy.ones(SearchAgents_no)
    ALG = numpy.random.randint(2, size= (SearchAgents_no, dim))
    fitness_array = numpy.zeros(SearchAgents_no)

    for i in range(0,SearchAgents_no):
        while numpy.sum(ALG[i, :]) == 0:
            ALG[i, :] = numpy.random.randint(2, size=(1, dim))
        fitness_array[i] = objf(ALG[i, :], trainInput, trainOutput, dim)

    min_fit = min(fitness_array)
    min_fit_index= numpy.argmin(fitness_array)
    best_ALG = ALG[min_fit_index,:].copy()
    best_fit = min_fit

    calculate_greatness(alg_size_matrix,fitness_array.copy())

    featurecount = 0
    for f in range(0, dim):
        if best_ALG[f] == 1:
            featurecount = featurecount + 1

    # Initialize convergence
    convergence_curve1 = numpy.zeros(iter)
    convergence_curve2 = numpy.zeros(iter)
    ############################
    s = solution()
    print("binary AAA is optimizing  \"" + objf.__name__ + "\"")
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    c=SearchAgents_no
    convergence_curve1[0] = best_fit
    convergence_curve2[0] = featurecount
    t=1
    while c < Max_iter:
         energy = calculate_energy(alg_size_matrix.copy())
         alg_friction = calculate_friction(alg_size_matrix.copy())
         for i in range(SearchAgents_no):
             istarve = 0
             while(energy[i] >= 0 and c < Max_iter):
                 neighbor=tournament_selection(fitness_array.copy())         #   Neigbour 1 ile 40 arasında bir tam sayı komşu alg
                 while neighbor==i :
                     neighbor=tournament_selection(fitness_array)
                 dimension_1=random.randint(0,dim-1)
                 dimension_2=random.randint(0,dim-1)
                 dimension_3=random.randint(0,dim-1)
                 while(dimension_1 == dimension_2 or dimension_1 == dimension_3 or dimension_2==dimension_3):
                     dimension_2=random.randint(0,dim-1)
                     dimension_3=random.randint(0,dim-1)

                 new_alg=ALG[i,:].copy()
                 new_alg[dimension_1] = new_alg[dimension_1] + (ALG[neighbor, dimension_1] - new_alg[dimension_1]) * (s_force - alg_friction[i]) * ((random.random() - 0.5) * 2)
                 new_alg[dimension_2] = new_alg[dimension_2] + (ALG[neighbor, dimension_2] - new_alg[dimension_2]) * (s_force - alg_friction[i]) *  math.cos(random.random() * 360)
                 new_alg[dimension_3] = new_alg[dimension_3] + (ALG[neighbor, dimension_3] - new_alg[dimension_3]) * (s_force - alg_friction[i]) *  math.sin(random.random() * 360)

                 for j in range(0, dim):
                     ss = transfer_functions_benchmark.s1(new_alg[j])
                     a = random.random()
                     if (a < ss ):
                         new_alg[j] = 1
                     else:
                         new_alg[j] = 0

                 while numpy.sum(new_alg) == 0:
                     new_alg = numpy.random.randint(2, size=(dim))
                 new_alg_fit = objf(new_alg, trainInput, trainOutput, dim)
                 energy[i] = energy[i] - e_loss/2

                 if( new_alg_fit < fitness_array[i] ):
                     ALG[i,:] = new_alg.copy()
                     fitness_array[i] = new_alg_fit
                     #labelsPred[i]= labelsPredYeni
                     istarve = 1
                 else:
                     energy[i] = energy[i] - e_loss/2

                 value = min(fitness_array)
                 index = numpy.argmin(fitness_array)
                 if value < best_fit:
                     best_fit = value
                     best_ALG = ALG[index, :].copy()
                     min_fit_index = index

                 featurecount = 0
                 for f in range(0, dim):
                     if best_ALG[f] == 1:
                         featurecount = featurecount + 1

                 c=c+1
                 if (c % SearchAgents_no == 0):
                     #print(['At iteration ' + str(c/SearchAgents_no) + ' the best fitness is ' + str(best_fit) + ' the best number of features: '+str(featurecount)])
                     convergence_curve1[t] = best_fit
                     convergence_curve2[t]  = featurecount
                     t = t + 1

             if istarve==0:
                starveAlg[i]= starveAlg[i] + 1


         #Evolution Process-----#Evolution Process-----#Evolution Process-----#
         calculate_greatness(alg_size_matrix, fitness_array.copy())
         rand_dim=random.randint(0,dim-1)
         minindex = numpy.argmin(alg_size_matrix)
         maxindex = numpy.argmax(alg_size_matrix)
         ALG[minindex,rand_dim] = ALG[maxindex,rand_dim]
         #Evolution Process----- #Evolution Process------#Evolution Process----#


         #Adaptation Process --  #Adaptation Process --#Adaptation Process
         index3 = numpy.argmax(starveAlg)
         if random.random() < ap :
             for i in range(dim):
                 ALG[index3, i] = ALG[index3, i] +  ( best_ALG[i] - ALG[index3,i] ) * random.random()
                 ss = transfer_functions_benchmark.s1(ALG[index3, i])
                 if (random.random() < ss):
                     ALG[index3, i] = 1;
                 else:
                     ALG[index3, i] = 0;
         # Adaptation Process -- # Adaptation Process -- # Adaptation Process -- #


         if (c % Max_iter == 0):
            print(['At iteration ' + str(c/SearchAgents_no) + ' the best fitness is ' + str(best_fit) + ' the best number of features: '+str(featurecount)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence1 = convergence_curve1
    s.convergence2 = convergence_curve2
    s.optimizer = "AAA"
    s.objfname = objf.__name__
    s.bestIndividual = best_ALG
    return s


def calculate_greatness(greatness, fitness_array1):

    max_val = max(fitness_array1)
    min_val = min(fitness_array1)

    for i in range(len(fitness_array1)):
        fitness_array1[i] = (fitness_array1[i]-min_val)  /  (max_val-min_val)
        fitness_array1[i] = 1 - fitness_array1[i]

    for i in range(len(greatness)):
        fKs = abs(greatness[i] / 2 )
        M = fitness_array1[i] / (fKs + fitness_array1[i])
        dX = M * greatness[i]
        greatness[i] = greatness[i] + dX


def calculate_energy(greatness):

    sorting = numpy.ones(len(greatness),'int')
    fGreat_surface = numpy.zeros(len(greatness))

    for i in range(0,len(greatness)):
        sorting[i] = i

    for i in range(len(greatness)-1):
        for j in range(i+1,len(greatness)):
            if(greatness[sorting[i]] > greatness[sorting[j]]):
                sorting[i] ,sorting[j] = sorting[j], sorting[i]
        fGreat_surface[sorting[i]] = i**2

    fGreat_surface[sorting[len(greatness)-1]] = (i+1)**2
    max_val = max(fGreat_surface)
    min_val = min(fGreat_surface)

    for i in range(len(fGreat_surface)):
        fGreat_surface[i] = (fGreat_surface[i] - min_val) / (max_val - min_val)

    return fGreat_surface


def calculate_friction(alg_size_matrix):

    fGreat_surface=numpy.zeros(len(alg_size_matrix))
    for i in range(len(alg_size_matrix)):
        r = ((alg_size_matrix[i] * 3) / (4* math.pi)) ** (1/3)
        fGreat_surface[i] = 2 * math.pi * (r**2)

    max_val = max(fGreat_surface)
    min_val =  min(fGreat_surface)

    for i in range(len(fGreat_surface)):
        fGreat_surface[i] = (fGreat_surface[i]-min_val)  /  (max_val-min_val)

    return fGreat_surface


def tournament_selection(fitness_array):

    individual1=random.randint(0,len(fitness_array)-1)
    individual2=random.randint(0,len(fitness_array)-1)

    while individual1==individual2:
        individual2=random.randint(0,len(fitness_array)-1)

    if (fitness_array[individual1] < fitness_array[individual2]):
        return individual1
    else:
        return individual2



