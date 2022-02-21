#!/usr/bin/env python
# coding: utf-8

# # One temperature
# The main classes of the code are:
# * $\textbf{Specie}$: contains specie data such as energy level and degeneracy and allows to compute the energy of a specie at a given temperature
# * $\textbf{Reaction}$ and $\textbf{sub-reaction}$: contain the reaction data such as reactants, products, exponents etc. and allows to compute the forward and backward coefficients
# * $\textbf{Problem}$: contains the problem information such as initial mixture composition, speed, temperature, etc. and allows to compute the mixture properties for a given state, solve the chemical relaxation problem and plot the results.
# 
# Each of these classes will be explained in details in the following pages.

# # Required packages and global constants
# 
# Firstly, some relevant python packages are imporoted and some useful constants are set. From the next page on the first lines will be hidden, they usually serve the same purpose as here and eventually add some tolerance parameter or load the results of the previous page.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy as cp
import sys
import scipy.optimize as opt
import scipy.integrate as itg


# In[2]:


Kb  = 1.3806503e-23;     # Boltzmann constant [J/K]
amu = 1.66053904020e-24; # 1/Avogadro's number [mol]
Ru = Kb/amu              # Universal gas constant [J/(K*mol)]

