# -*- coding: utf-8 -*-
"""DNAchangeSimulation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14Z2reWfxUGmUeP_9_8RAZNxu88i8DW7g

Libraries
"""

from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.linalg import expm

"""Function for generationg the ancestor DNA"""

def generate_ancestor(n, pi=None):
    nucleotides = [0, 1, 2, 3]  # T, C, A, G

    if pi is None:
        # If no probability distribution is provided, use a random distribution, equal probabilities of 0.25
        return list(random.choices(nucleotides, k=n))
    else:
        # If a custom probability distribution is provided, use it
        if len(pi) != 4:
            raise ValueError("Probability distribution must contain exactly 4 values.")

        # Check if the probabilities sum to 1
        if abs(sum(pi) - 1) > 1e-10:
            raise ValueError("Probabilities must sum to 1.")

        return list(random.choices(nucleotides, weights=pi, k=n))

"""Function to generate rate matrix"""

def generate_rate_matrix():
    matrix = np.zeros((4, 4))
    for i in range(4):
        row_sum = 0 # sum of non-diagonal elements of current row
        for j in range(4):
            if i != j: # fill up non-diagonal elements of current row
                element = np.random.uniform(0.01, 1.0)  # Non-diagonal elements are between 0.01 and 1.0
                row_sum += element
                matrix[i, j] = element
        matrix[i,i] = -row_sum # Ensure every row adds up to 0

    return matrix

"""Function for transition matrix"""

def transition_matrix(Q, t):
    Qt = np.dot(Q, t)
    P = expm(Qt)
    return P

"""Function for waiting time generation(1st Approach)"""

def initialize_waiting_times(DNA_seq, Q_dict):
    n_bases = len(DNA_seq)
    waiting_times = np.full((n_bases, 4), float('inf'))  # Initialize all waiting times to infinity

    # Iterate through each possible nucleotide transition
    for next_base in range(4):
        rates = np.array([Q_dict[curr_base, next_base] if curr_base != next_base else float('inf')
                          for curr_base in DNA_seq])

        # Calculate waiting times where rates are not infinity
        valid_rates = rates != float('inf')
        waiting_times[valid_rates, next_base] = np.random.exponential(scale=1.0 / rates[valid_rates])

    # Find the minimum time and its position
    min_position = np.unravel_index(np.argmin(waiting_times), waiting_times.shape)
    min_time = waiting_times[min_position]
    return min_position, min_time

def update_waiting_times(DNA_seq, Q_dict, min_position, min_time):
    seq_index, new_base = min_position

    DNA_seq[seq_index] = new_base # Substitue with new base in DNA sequence

    # Regenerate all waiting times
    min_position, min_time = initialize_waiting_times(DNA_seq, Q_dict)

    return min_position, min_time

def simulate_seq(max_time, rate_matrix, ancestor_seq = None):
    history = [ancestor_seq.copy(),]

    time_passed = 0

    DNA_seq = ancestor_seq.copy()
    min_position, min_time = initialize_waiting_times(DNA_seq, rate_matrix)

    while time_passed + min_time <= max_time:
        time_passed += min_time
        seq_index, new_base = min_position
        min_position, min_time = update_waiting_times(DNA_seq, rate_matrix, min_position, min_time)
        DNA_seq[seq_index] = new_base
        history.append(DNA_seq.copy())

    return history[:-1]

def average_substitution(Q, t, repeats, length, pi):

    n = length
    ns_per_site_list = []
    ns_total_list = []
    for i in range(repeats):
        ancestor_sequence = generate_ancestor(length, pi)
        history = simulate_seq(t, Q, ancestor_sequence)
        ns_total = len(history)-1
        ns_per_site = ns_total/n
        ns_per_site_list.append(ns_per_site)
        ns_total_list.append(ns_total)

    average_ns_total = np.average(ns_total_list)
    average_ns_per_site = average_ns_total/n

    return ns_per_site_list, average_ns_per_site

"""Function for waiting time generation (2nd Approach)"""

def initialize_first_waiting_time(DNA_seq, Q_dict):
    n = len(DNA_seq)
    sum = 0
    for i in range(n):
      sum = sum -Q_dict[DNA_seq[i],DNA_seq[i]]

    first_waiting_time = np.random.exponential(1.0/sum)
    return first_waiting_time

def generate_probabilty(DNA_seq, Q_dict):
    n = len(DNA_seq)
    sum = 0
    for i in range(n):
      sum = sum -Q_dict[DNA_seq[i],DNA_seq[i]]

    p = np.zeros(n)
    for i in range(n):
      p[i] = -(Q_dict[DNA_seq[i],DNA_seq[i]]/sum)

    return p

def new_dna_seq(DNA_seq, p):
    n = len(DNA_seq)
    new_DNA_seq = np.random.choice(4, size=n, p=p)
    return new_DNA_seq

"""Function  to update the waiting times"""

def update_waiting_times(DNA_seq, Q_dict, min_position, min_time):
    seq_index, new_base = min_position

    DNA_seq[seq_index] = new_base # Substitue with new base in DNA sequence

    # Regenerate all waiting times
    min_position, min_time = initialize_first_waiting_time(DNA_seq, Q_dict)

    return min_position, min_time