#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import relevant packages

import pandas as pd
import numpy as np


# In[2]:


# Set global variables
global R 
R  = 8314
global Kb 
Kb = 1.380649*1e-23


# In[3]:


# General class definition

class trve_variable():
    def __init__(self):
        self.t = lambda T : []    # Definition always missing below
        self.r = lambda T : []
        self.v = lambda T : []
        self.e = lambda T : []
        self.tot = lambda T : []
class specie():
    'This class contains specie data'
    def __init__(self):
        
        # Dati delle tabelle di Zanardi ref[1]
        
        self.M  = []
        self.A  = []
        self.B  = []
        self.C  = []
        self.I  = []
        self.h  = []
        self.Ov = []
        self.Or = []
        self.Ds = []
        self.electro = pd.DataFrame({'Level': [],
                                     'g'    : [],
                                     'Oe'   : []
                                    })
        self.Z = trve_variable()
        self.E = trve_variable()
        self.CV= trve_variable()
    
    # See equilibrium solutions of WCU model and the partition function Week 5 2021 Hypersonic frezzotti
    def updateR(self):
        'Updates thermodynamic variable R'
        self.R = R/self.M
    
    def updateZ(self):
        'Updates partition functions'
        self.Z.r = lambda T : T/self.Or  # Approximated
        self.Z.v = lambda T : 1/(1-np.exp(-self.Ov/T)) # Armonic oscillator
        self.Z.e = lambda T : np.sum(self.electro['g']*np.exp(-self.electro['Oe']/T))
        
        self.Z.tot = lambda T : self.Z.r(T) + self.Z.v(T) + self.Z.e(T) # Sure it is not a product ?
        
        
    def updateE(self):
        'Updates energy functions'
        self.E.r = lambda T : self.R*T
        self.E.v = lambda T : self.Ov/(np.exp(self.Ov/T) - 1) # Check n kb and rho
        self.E.e = lambda T : self.R*1/self.Z.e(T)*np.sum(                                 self.electro['g']*self.electro['Oe']*                                 np.exp(-self.electro['Oe']/T))
        
        self.E.tot = lambda T : self.E.r(T) + self.E.v(T) + self.E.e(T)
        
    def updateCV(self):
        'Updates Cv'
        self.CV.r = lambda T : self.R
        # No dissociation correction / no anharmonic correction
        self.CV.v = lambda T : rho*R*(self.Ov/T)^2*                                 np.exp(self.Ov/T) /                                 (np.exp(self.Ov/T)-1)^2 # Where does rho come from?
        self.CV.e = lambda T : self.R/T^2 * (1/self.Z.e(T)*                                 np.sum(self.electro['g'] * self.electro['Oe']^2 *                                np.exp(-self.electro['Oe']/T) -                                 self.E.e(T)/self.R))
        
        self.CV.tot = lambda T : self.CV.r(T) + self.CV.v(T) + self.CV.e(T)        
        
    def update(self):
        'To be run once the thermo constant are set'
        self.updateR()
        self.updateZ()
        self.updateE()
        self.updateCV()


# In[4]:


#N2 definition

N2 = specie()

N2.M  = 28.0134
N2.A  = 2.68e-2
N2.B  = 3.18e-1
N2.C  = -1.13e1
N2.I  = 5.36498e7
N2.h  = 0
N2.Ov = 3395
N2.Or = 2.9
N2.Ds = 3.36e7
N2.update()

N2.electro = {'Level': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
              'g'    : [1, 3, 6, 6, 3, 1, 2, 2, 5, 1, 6, 6, 10, 6, 6],
           'Oe': [0, 7.2231565e+04, 8.5778626e+04, 8.6050267e+04, 
                  9.5351186e+04, 9.8056357e+04, 9.9682677e+04, 
                  1.0489765e+05, 1.1164896e+05, 1.2258365e+05,
                  1.2488569e+05, 1.2824762e+05, 1.3380609e+05, 
                  1.4042964e+05, 1.5049589e+05]
          }
N2.electro = pd.DataFrame(N2.electro)

N2.electro # Print electro table in pandas format

N2.Z.r(500)


# In[5]:


#O2 definition

O2 = specie()

O2.M  = 31.9988
O2.A  = 4.49e-02
O2.B  = -8.26e-02
O2.C  = -9.20
O2.I  = 3.63832e+07
O2.h  = 0
O2.Ov = 2239
O2.Or = 2.1
O2.Ds = 1.54e+07
O2.update()

O2.electro = {'Level': [0, 1, 2, 3, 4, 5, 6], 
           'g': [3, 2, 1, 1, 6, 3, 3],
           'Oe': [0, 1.1391560e+04, 1.8984739e+04,
                 4.7559736e+04, 4.9912421e+04, 5.0922686e+04,
                 7.1898633e+04]
          }
O2.electro = pd.DataFrame(N2.electro)

O2.electro # Print electro table in pandas format


# In[6]:


#NO definition

NO = specie()

NO.M  = 30.0061
NO.A  = 4.36e-02
NO.B  = -3.36e-02
NO.C  = -9.58
NO.I  = 2.97808e+07
NO.h  = 3.0091e+06
NO.Ov = 2817
NO.Or = 2.5
NO.Ds = 2.09e+07
NO.update()

NO.electro = {'Level': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
           'g': [4, 8, 2, 4, 4, 4, 4, 2, 4, 2, 4, 4, 2, 2, 2, 4],
           'Oe': [0, 5.4673458e+04, 6.3171396e+04,
                 6.5994503e+04, 6.9061210e+04, 7.0499985e+04,
                 7.4910550e+04, 7.6288753e+04, 8.6761885e+04,
                 8.7144312e+04, 8.8860771e+04, 8.9817556e+04,
                 8.9884459e+04, 9.0427021e+04, 9.0642838e+04,
                 9.1117633e+04]
          }
NO.electro = pd.DataFrame(NO.electro)

NO.electro # Print electro table in pandas format


# In[7]:


#N definition

N = specie()

N.M  = 14.0067
N.A  = 1.16e-02
N.B  = 6.03e-01
N.C  = -1.24e+01
N.I  = 1.00090e+08
N.h  = 3.3747e+07
N.Ov = 0
N.Or = 0
N.Ds = 0
N.update()

N.electro = {'Level': [0, 1, 2], 
           'g': [4, 10, 6],
           'Oe': [0, 2.7664696e+04, 4.1493093e+04]
          }
N.electro = pd.DataFrame(N.electro)

N.electro # Print electro table in pandas format


# In[8]:


#O definition

O = specie()

O.M  = 15.9994
O.A  = 2.03e-02
O.B  = 4.29e-01
O.C  = -1.16e+01
O.I  = 8.21013e+07
O.h  = 1.5574e+07
O.Ov = 0
O.Or = 0
O.Ds = 0
O.update()

O.electro = {'Level': [0, 1, 2, 3, 4], 
           'g': [5, 3, 1, 5, 1],
           'Oe': [0, 2.2770776e+02, 3.2656888e+02,
                  2.2830286e+04, 4.8619930e+04]
          }
O.electro = pd.DataFrame(O.electro)

O.electro # Print electro table in pandas format


# In[9]:


#N2p definition

N2p = specie

N2p.M  = 28.0128514
N2p.A  = 2.68e-02
N2p.B  = 3.18e-01
N2p.C  = -1.13e+01
N2p.I  = 0
N2p.h  = 5.3886e+07
N2p.Ov = 3395
N2p.Or = 2.9
N2p.Ds = 3.00e+07

N2p.electro = {'Level': [0, 1, 2, 3, 4,5 ,6 ,7 ,8 ,9 ,10, 11, 12, 13, 14, 15, 16], 
           'g': [2, 4, 2, 4,8, 8, 4, 4, 4, 4, 8, 8, 4, 4, 2, 2, 4],
           'Oe': [0, 1.3189972e+04, 3.6633231e+04,
                  3.6688768e+04, 5.9853048e+04, 6.6183659e+04,
                  7.5989919e+04, 7.6255086e+04, 8.2010186e+04, 
                  8.4168349e+04, 8.6326512e+04, 8.9204062e+04,
                  9.2081613e+04, 9.2225490e+04, 9.2937684e+04, 
                  9.6397938e+04, 1.0359181e+05]
          }
N2p.electro = pd.DataFrame(N2p.electro)

N2p.electro # Print electro table in pandas format


# In[10]:


#O2p definition

O2p = specie()

O2p.M  = 31.9982514
O2p.A  = 4.49e-02
O2p.B  = -8.26e-02
O2p.C  = -9.20
O2p.I  = 0
O2p.h  = 3.6592e+07
O2p.Ov = 2239
O2p.Or = 2.1
O2p.Ds = 2.01e+07
O2p.update()

O2p.electro = {'Level': [0, 1, 2, 3, 4,5 ,6 ,7 ,8 ,9 ,10, 11, 12, 13, 14], 
           'g': [4, 8, 4, 6, 4, 2, 4, 4, 4, 4, 8, 4, 2, 2, 4],
           'Oe': [0, 4.7354408e+04, 5.8373987e+04, 
                 5.8414273e+04, 6.2298966e+04, 6.7334679e+04,
                 7.1219372e+04, 7.6542841e+04, 8.8196920e+04,
                 8.8196920e+04, 9.4239776e+04, 9.4959163e+04,
                 9.5920265e+04, 9.9850999e+04, 1.0359181e+05]
          }
O2p.electro = pd.DataFrame(O2p.electro)

O2p.electro # Print electro table in pandas format


# In[11]:


#NOp definition

NOp = specie()

NOp.M  = 30.0055514
NOp.A  = 3.02e-01
NOp.B  = -3.50
NOp.C  = -3.74
NOp.I  = 0
NOp.h  = 3.3000e+07
NOp.Ov = 2817
NOp.Or = 2.5
NOp.Ds = 3.49e+07
NOp.update()

NOp.electro = {'Level': [0, 1, 2, 3, 4,5 ,6 ,7], 
           'g': [1, 3, 6, 6, 3, 1, 2, 2],
           'Oe': [0, 7.5089678e+04, 8.5254624e+04, 
                 8.9035726e+04, 9.7469826e+04, 1.0005530e+05,
                 1.0280337e+05, 1.0571386e+05]
          }
NOp.electro = pd.DataFrame(NOp.electro)

NOp.electro # Print electro table in pandas format


# In[12]:


#Np definition

Np = specie()

Np.M  = 14.0061514
Np.A  = 1.16e-02
Np.B  = 6.03e-01
Np.C  = -1.24e+01
Np.I  = 0
Np.h  = 1.3438e+08
Np.Ov = 0
Np.Or = 0
Np.Ds = 0
Np.update()

Np.electro = {'Level': [0, 1, 2, 3, 4, 5 ,6], 
           'g': [1, 3, 5, 5, 1, 5, 15],
           'Oe': [0, 7.0068352e+01, 1.8819180e+02,
                  2.2036569e+04, 4.7031835e+04, 6.7312522e+04,
                  1.3271908e+05]
          }
Np.electro = pd.DataFrame(Np.electro)

Np.electro # Print electro table in pandas format


# In[13]:


#Op definition

Op = specie()

Op.M  = 15.9988514
Op.A  = 2.03e-02
Op.B  = 4.29e-01
Op.C  = -1.16e+01
Op.I  = 0
Op.h  = 9.8056e+07
Op.Ov = 0
Op.Or = 0
Op.Ds = 0
Op.update()

Op.electro = {'Level': [0, 1, 2], 
           'g': [4, 10,6],
           'Oe': [0, 3.8583347e+04, 5.8223492e+04]
          }
Op.electro = pd.DataFrame(Op.electro)

Op.electro # Print electro table in pandas format


# In[14]:


#em definition

em = specie()

em.M  = 0.00054858
em.A  = 0
em.B  = 0
em.C  = -1.20e+01
em.I  = 0
em.h  = 0
em.Ov = 0
em.Or = 0
em.Ds = 0
em.update()

em.electro = {'Level': [0], 
           'g': [1],
           'Oe': [0]
          }
em.electro = pd.DataFrame(em.electro)

em.electro # Print electro table in pandas format


# In[15]:


# Pandas usefull commands
# Show a column of the table
print(O2.electro['Oe'])

print('\n\n')
# Access a variable
print(' The requested variable is ' , O2.electro['Oe'][2])

