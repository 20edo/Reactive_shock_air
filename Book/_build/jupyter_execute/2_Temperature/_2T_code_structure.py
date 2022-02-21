#!/usr/bin/env python
# coding: utf-8

# # Two-temperature
# The main classes of the code are:
# * $\textbf{Specie}$: contains specie data such as energy level and degeneracy and allows to compute the energy of a specie at a given temperature
# * $\textbf{Reaction and sub-reaction}$: contain the reaction data such as reactants, products, exponents etc. and allows to compute the forward and backward coefficients
# * $\textbf{Problem}$: contains the problem information such as initial mixture composition, speed, temperature, etc. and allows to compute the mixture properties for a given state, solve the chemical relaxation problem and plot the results.
# 
# Each of these classes will be explained in details in the following pages.

# # Two temperature
# The two temperature extends the classes prepared for the one temperature mixture model distinguishing the effects of roto-translational and vibro-electronic temperatures which represent two different flow variables. <br>
# While in general the energy is a combination of the four energy which thermalize differently, the approximation of tranlational-rotational and vibro-electronic temperatures is common since the themalization of each couple of degrees of freedom. <br>

# ```{note}
# Appropriate _2T functions are defined just where needed. The correspective one temperature version are still present in general.
# ```

# # Required packages and global constants
# 
# Firstly, some relevant python packages are imporoted and some useful constants are set. From the next page on the first lines will be hidden, they usually serve the same purpuse as here and eventually add some tolerance parameter or load the results of the previous page.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy as cp
import sys
import scipy.optimize as opt
import scipy.integrate as itg


# The function "find letter" is defined in order to distinguish the differents chemical reaction considered and compute the corresponding forward and backward reaction coefficient. See [Chemical reaction class definition](#Chemical-reaction-class-definition)

# In[2]:


def find_letter(letter, lst):
    return any(letter in word for word in lst)


# In[3]:


Kb  = 1.3806503e-23;     # Boltzmann constant [J/K]
amu = 1.66053904020e-24; # 1/Avogadro's number [mol]
Ru = Kb/amu              # [J/(K*mol)]

# I set minimum and maximum values for the equilibrium constant Kc 
# in order to avoid convergence problem due to too big or too small numerical values
kc_min = 1e-20
kc_max = 1e20


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.figsize"] = (15,10)

rtol = 1e-12
atol = 1e-12
debug = 0

