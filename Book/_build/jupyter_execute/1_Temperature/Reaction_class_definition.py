#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'Specie_class_definition.ipynb')


# # Reaction and sub-reaction classes
# Two classes for the modeling of the chemical reactions have been defined:
# 
# 1. The dissociation reactions are grouped in classes made by many sub-reactions where the difference between them is the inert molecule triggering the process.
# 
# 2. On the other hand some chemical reactions are written as single sub-reaction.
# 
# While there is no real need for this division, this worklow allows to group together the reactions that somehow "do the same thing" to be able to include or exclude one of them by just a line of code. Despite that, all reactions may be re-written to be one or many variables of class sub-reaction with no effect on the code execution.
# 
# The definition of the parameters used to model the two classes is a little tricky and may be more user-friendly.

# # Sub-reaction class definition
# A subreaction is defined by the soichometric coefficients, the reactants, the products and the reaction rate coefficients. The latter are computed as follows:
# * Forward reaction coefficient : &nbsp; &nbsp; &nbsp; $k_f$ is expressed as $k_f = C_f T ^{\eta_f}e^{\frac{\theta_d}{T}}$ (coefficients from Park's tables [1])
# * Equilibrium reaction coefficient : &nbsp; $k_c$ is computed from the polynomial interpolation by Park [1]
# * Backward reaction coefficient: &nbsp; &nbsp; &nbsp; $k_b$ is computed as $k_b = \frac{k_f}{k_c}$

# In[2]:


class subreaction:
    def __init__(self, reactants, products, stoichr, stoichp, Cf, nf, th_d):
        self.reactants = reactants
        self.products  = products
        self.stoichr   = stoichr # Stoichometric coefficient of the reactants
        self.stoichp   = stoichp # Stoichometric coefficient of the products
        self.Cf        = Cf
        self.nf        = nf
        self.th_d      = th_d
        self.A         = None # Coefficients of the polynomial used to compute Kc
        self.e_mol     = None

        
    def kf(self, T):
        kf = self.Cf * T ** self.nf * np.exp(-self.th_d / T)
        return kf
        
    def kb(self, T):
        kb = self.kf(T)/self.kc(T)
        return kb
    
    def kc(self, T):
        Z = 10000 / T 
        exponent = self.A[0] + self.A[1]*Z + self.A[2]*Z**2 + self.A[3]*Z**3 + self.A[4]*Z**4
        
        if exponent < np.log(kc_min):
            kc = kc_min
        elif exponent > np.log(kc_max):
            kc = kc_max
        else :
            kc = np.exp(exponent)
        return kc


# # Reaction class definition
# A reaction is a list of sub-reactions that is defined according to what explained above. However, since the coefficient of the polynomial used to compute the equilibrium reaction coefficient do not depend on the impact body, they do not need to be repeated for every sub-reaction but are stored in the reaction class and automatically updated in each newly created sub-reaction.

# In[3]:


class reaction:
    def __init__(self, A, e_mol):
        self.subreactions = []
        self.A            = np.array(A)
        self.e_mol        = e_mol
    
    
    def add_subreaction(self, subr):
        subr.A     = self.A
        subr.e_mol = self.e_mol
        self.subreactions.append(subr)


# # Reaction variables definition
# The 47 reactions for the 11 species air model are defined below, data are extracted from Park's tables [1].

# In[4]:


# The pre-exponential coefficent Cf is devided by 1e6 to convert from cm3/mol to m3/mol
# e_mol is multiplicated by 4184 in order to convert Kcal/mol in J/mol

O2diss = reaction([1.335, -4.127, -0.616, 0.093, -0.005], -117.98*4184)
O2diss.add_subreaction(subreaction([O2, N], [O, N], [1, 1], [2,1], 8.25e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, O], [O, O], [1, 1], [2,1], 8.25e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, Np], [O, Np], [1, 1], [2,1], 8.25e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, Op], [O, Op], [1, 1], [2,1], 8.25e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, N2], [O, N2], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, O2], [O, O2], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, NO], [O, NO], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, N2p], [O, N2p], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, O2p], [O, O2p], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, NOp], [O, NOp], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss.add_subreaction(subreaction([O2, em], [O, em], [1, 1], [2,1], 1.32e22/1e6,
                                  -1, 59500))


# In[5]:


N2diss = reaction([ 3.898, -12.611, 0.683, -0.118, 0.006], -225.00*4184)
N2diss.add_subreaction(subreaction([N2, N], [N, N], [1, 1], [2,1], 1.11e22/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, O], [N, O], [1, 1], [2,1], 1.11e22/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, Np], [N, Np], [1, 1], [2,1], 1.11e22/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, Op], [N, Op], [1, 1], [2,1], 1.10e22/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, N2], [N, N2], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, O2], [N, O2], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, NO], [N, NO], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, N2p], [N, N2p], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, O2p], [N, O2p], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, NOp], [N, NOp], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss.add_subreaction(subreaction([N2, em], [N, em], [1, 1], [2,1], 1.11e24/1e6,
                                  -1.6, 113200))


# In[6]:


NOdiss = reaction([1.549, -7.784, 0.228, -0.043, 0.002], -150.03*4184) 
NOdiss.add_subreaction(subreaction([NO, N], [N, O], [1, 1], [2, 1], 4.6e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, O], [N, O], [1, 1], [1, 2], 4.6e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, Np], [N, O, Np], [1, 1], [1, 1, 1], 4.6e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, Op], [N, O, Op], [1, 1], [1, 1, 1], 4.6e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, N2], [N, O, N2], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, O2], [N, O, O2], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, NO], [N, O, NO], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, N2p], [N, O, N2p], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, O2p], [N, O, O2p], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, NOp], [N, O, NOp], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss.add_subreaction(subreaction([NO, em], [N, O, em], [1, 1], [1, 1, 1], 7.36e19/1e6,
                                  -0.5, 75500))


# In[7]:


#Exchange reactions
NO_O       = subreaction([NO, O], [N, O2], [1, 1], [1, 1], 2.16e8/1e6, 1.29, 19220)
NO_O.A     = np.array([0.215, -3.657, 0.843, -0.136, 0.007])
NO_O.e_mol = -32.05*4184

O_N2       = subreaction([O, N2], [N, NO], [1, 1], [1, 1], 3.18e13/1e6, 0.1, 37700)
O_N2.A     = np.array([2.349, -4.828, 0.455, -0.075, 0.004])
O_N2.e_mol = -74.97*4184

O_O2p      = subreaction([O, O2p], [O2, Op], [1, 1], [1, 1], 6.85e13/1e6, -0.520, 18600)
O_O2p.A    = np.array([-0.411, -1.998, -0.002, 0.005, 0.00])
O_O2p.e_mol= -36.88*4184

N2_Np       = subreaction([N2, Np], [N, N2p], [1, 1], [1, 1], 9.85e12/1e6, -0.180, 12100)
N2_Np.A     = np.array([1.963, -3.116, 0.692, -0.103, 0.005])
N2_Np.e_mol = -24.06*4184

O_NOp      = subreaction([O, NOp], [NO, Op], [1, 1], [1, 1], 2.75e13/1e6, 0.010, 51000)
O_NOp.A    = np.array([1.705, -6.223, 0.522, -0.090, 0.005])
O_NOp.e_mol= -101.34*4184

N2_Op       = subreaction([N2, Op], [O, N2p], [1, 1], [1, 1], 6.33e16/1e6, -0.210, 22200)
N2_Op.A     = np.array([2.391, -2.443, -0.080, 0.027, -0.002])
N2_Op.e_mol = -44.23*4184

N_NOp       = subreaction([N, NOp], [NO, Np], [1, 1], [1, 1], 2.21e15/1e6, -0.020, 61100)
N_NOp.A     = np.array([2.132, -5.550, -0.249, 0.041, -0.002])
N_NOp.e_mol = -121.51*4184

O2_NOp      = subreaction([O2, NOp], [NO, O2p], [1, 1], [1, 1], 1.03e16/1e6, -0.170, 32400)
O2_NOp.A    = np.array([2.115, -4.225, 0.524, -0.095, 0.005])
O2_NOp.e_mol= -64.46*4184

NOp_N       = subreaction([NOp, N], [N2p, O], [1, 1], [1, 1], 1.7e13/1e6, 0.400, 35500)
NOp_N.A     = np.array([1.746, -3.838, -0.013, 0.013, -0.001])
NOp_N.e_mol = -70.60*4184

# Associative ionization
O___N       = subreaction([O, N], [NOp, em], [1, 1], [1, 1], 1.53e11/1e6, -0.370, 32000)
O___N.A     = np.array([-6.234, -5.536, 0.494, -0.058, 0.003])
O___N.e_mol = -63.69*4184

O___O       = subreaction([O, O], [O2p, em], [1, 1], [1, 1], 3.85e11/1e6, 0.490, 80600)
O___O.A     = np.array([-3.904, -13.418, 1.861, -0.288, 0.015])
O___O.e_mol = -160.20*4184

N___N       = subreaction([N, N], [N2p, em], [1, 1], [1, 1], 1.79e11/1e6, 0.770, 67500)
N___N.A     = np.array([-4.488, -9.374, 0.481, -0.044, 0.002])
N___N.e_mol = -134.29*4184

# Electron impact ionization
O_ion       = subreaction([O, em], [Op, em], [1, 1], [1, 2], 3.9e33/1e6, -3.780, 158500)
O_ion.A     = np.array([-2.980, -19.534, 1.244, -0.190, 0.010])
O_ion.e_mol = -315.06*4184

N_ion       = subreaction([N, em], [Np, em], [1, 1], [1, 2], 2.5e34/1e6, -3.820, 168600)
N_ion.A     = np.array([-2.553, -18.870, 0.472, -0.060, 0.003])
N_ion.e_mol = -335.23*4184


# While sub-reactions are somehow general, reactions include the possible impact with all other species, thus the sub-reactions to be included in each reaction depend on the mixture model considered. In the following, the reactions for the 7 specie air model are defined.

# In[8]:


O2diss_7s = reaction([1.335, -4.127, -0.616, 0.093, -0.005], -117.98*4184)
O2diss_7s.add_subreaction(subreaction([O2, N], [O, N], [1, 1], [2,1], 8.25e19/1e6,
                                  -1, 59500))
O2diss_7s.add_subreaction(subreaction([O2, O], [O, O], [1, 1], [2,1], 8.25e19/1e6,
                                  -1, 59500))
O2diss_7s.add_subreaction(subreaction([O2, N2], [O, N2], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss_7s.add_subreaction(subreaction([O2, O2], [O, O2], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss_7s.add_subreaction(subreaction([O2, NO], [O, NO], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss_7s.add_subreaction(subreaction([O2, NOp], [O, NOp], [1, 1], [2,1], 2.75e19/1e6,
                                  -1, 59500))
O2diss_7s.add_subreaction(subreaction([O2, em], [O, em], [1, 1], [2,1], 1.32e22/1e6,
                                  -1, 59500))


# In[9]:


N2diss_7s = reaction([ 3.898, -12.611, 0.683, -0.118, 0.006], -225.00*4184)
N2diss_7s.add_subreaction(subreaction([N2, N], [N, N], [1, 1], [2,1], 1.11e22/1e6,
                                  -1.6, 113200))
N2diss_7s.add_subreaction(subreaction([N2, O], [N, O], [1, 1], [2,1], 1.11e22/1e6,
                                  -1.6, 113200))
N2diss_7s.add_subreaction(subreaction([N2, N2], [N, N2], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss_7s.add_subreaction(subreaction([N2, O2], [N, O2], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss_7s.add_subreaction(subreaction([N2, NO], [N, NO], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss_7s.add_subreaction(subreaction([N2, NOp], [N, NOp], [1, 1], [2,1], 3.7e21/1e6,
                                  -1.6, 113200))
N2diss_7s.add_subreaction(subreaction([N2, em], [N, em], [1, 1], [2,1], 1.11e24/1e6,
                                  -1.6, 113200))


# In[10]:


NOdiss_7s = reaction([1.549, -7.784, 0.228, -0.043, 0.002], -150.03*4184) 
NOdiss_7s.add_subreaction(subreaction([NO, N], [N, O], [1, 1], [2, 1], 4.6e17/1e6,
                                  -0.5, 75500))
NOdiss_7s.add_subreaction(subreaction([NO, O], [N, O], [1, 1], [1, 2], 4.6e17/1e6,
                                  -0.5, 75500))
NOdiss_7s.add_subreaction(subreaction([NO, N2], [N, O, N2], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss_7s.add_subreaction(subreaction([NO, O2], [N, O, O2], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss_7s.add_subreaction(subreaction([NO, NO], [N, O, NO], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss_7s.add_subreaction(subreaction([NO, NOp], [N, O, NOp], [1, 1], [1, 1, 1], 2.3e17/1e6,
                                  -0.5, 75500))
NOdiss_7s.add_subreaction(subreaction([NO, em], [N, O, em], [1, 1], [1, 1, 1], 7.36e19/1e6,
                                  -0.5, 75500))

