#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', '1T_code_structure.ipynb')


# # Specie class definition
# 
# For each specie a common framework is defined which allows to compute thermodynamic derivatives, energy, temperature and other relevant values for the subsequent calculations starting from the data provided by :cite:p:`park_convergence_1985` and \{cite:p}`Zanardi_2020`.
# 
# See :cite:t:`1987:nelson` for an introduction to non-standard analysis.See :cite:t:`1987:nelson` for an introduction to non-standard analysis. \textcolor{red}{Le citazioni non funzionano}
# 
# Assuming that the different contributions can be computed indipendently, the temperature and CVs are defined as the sum of the translational-rotational, vibrational and electronic which are calculated as:
# 
# * Translational-rotational : 
#     1. If the molecule is monoatomic, it has no rotational dofs, then $ CV^t = \frac{3}{2} R_{gas} T $
#     2. If the molecule is not monoatomic, the rotational contribution must be added $ CV^{(rt)} = CV^t + R $
#     
#     Then, energy is computed as $e^{tr} = CV  T$.
#     
# * Vibrational :
#     The contribution of vibrational dofs to energy is computed by assuming armhonic oscillator potential. The energy can becomes: $e^{v} = R_{gas} \frac{\theta_v}{e^{\frac{\theta_v}{T}-1}} $ <br>
#     The analytical expression for the heat is computed by deriving with respect to T.
#     
# * Electronic :
#     Given the partition function (Zanardi's tables), the energy due to electronic energy levels is computed as: $ e = R_{gas}T^2 \frac{\partial}{\partial T} log(Z) = \frac{R_gas}{Z} \sum_i g_i \theta_i e^{-\frac{\theta_i}{T}}$.
#     
#     Analogously with the vibrational contribution the expression for the specific heat is obtained deriving the formula with respect to T.
#     
# All energies are written as sensible energy whose reference temperature is $T_0 = 298.15 K$

# In[2]:


class specie:
    def __init__(self, mmol, th_v, th_r, th_d, h_form = 0):
        self.th_v = th_v
        self.th_r = th_r
        self.th_d = th_d
        self.electro_g = None
        self.electro_th = None
        self.e0 = None
        self.e_rot = None
        self.e_vib = None
        self.e_ele = None
        self.e = None
        self.mmol = mmol*1e-3
        self.R = Kb/(mmol*1e-3)/amu
        self.h_form = h_form
        specie.name = None
        
    def energy(self, T):
        'Computes the energy and Cv as the sum of translational-rotational, vibrational and electronic contributions'
        e = 0
        cv = 0
        T0 = 298.15
        
        # Traslational - rotational 
        if self.th_r > 0:
            cv_rot = 3/2 * self.R + self.R
        else:
            cv_rot = 3/2 * self.R
            
        e_rot  = cv_rot * (T-T0)
        
        # Vibrational 
        if self.th_v > 0:
            cv_vib = self.R*((self.th_v/T)**2)*(np.exp(self.th_v/T))/((np.exp(self.th_v/T)-1)**2)
            e_vib  = self.R*self.th_v/(np.exp(self.th_v/T) - 1) - self.R*self.th_v/(np.exp(self.th_v/T0) - 1)
        else:
            cv_vib = 0
            e_vib  = 0
        
        # Electronic
        Z  = np.sum(self.electro_g * np.exp(-self.electro_th/T))
        Z0 = np.sum(self.electro_g * np.exp(-self.electro_th/T0))
        Z1 = np.sum(self.electro_g * np.exp(-self.electro_th/T) * self.electro_th)
        Z10= np.sum(self.electro_g * np.exp(-self.electro_th/T0) * self.electro_th)
        Z2 = np.sum(self.electro_g * np.exp(-self.electro_th/T) * self.electro_th**2) 
        Z20= np.sum(self.electro_g * np.exp(-self.electro_th/T0) * self.electro_th**2) 
        e_ele  = self.R * Z1 / Z - self.R * Z10 / Z0
        cv_ele = self.R/(T**2)*(-(e_ele/self.R)**2 + Z2 / Z)
        
        # Update Cv and e values
        cv = cv_rot + cv_vib + cv_ele
        e  = e_rot  + e_vib  + e_ele 
        e  = e - self.R * T0                         # Additional term to transform in sensible energy
        e  = e + self.h_form
        
        return e, cv


# # Specie variables definition
# Follows the definition of a number of selected species. The data used can be found in {% cite Zanardi_2020 %}.

# In[3]:


N2 = specie(28.0134, 3395, 2.9, 113200)
N2.electro_g  = np.array([1, 3, 6, 6, 3, 1, 2, 2, 5, 1, 6, 6, 10, 6, 6])
N2.electro_th = np.array([0.0, 7.2231565e+04, 8.5778626e+04, 8.6050267e+04, 9.5351186e+04, 9.8056357e+04,
                          9.9682677e+04, 1.0489765e+05, 1.1164896e+05, 1.2258365e+05, 1.2488569e+05, 
                          1.2824762e+05, 1.3380609e+05, 1.4042964e+05, 1.5049589e+05])
N2.name = 'N2'

O2 = specie(31.9988, 2239.0, 2.1, 59500)
O2.electro_g = np.array([3, 2, 1, 1, 6, 3, 3])
O2.electro_th= np.array([0, 1.1391560e+04, 1.8984739e+04, 4.7559736e+04, 4.9912421e+04, 5.0922686e+04,
                         7.1898633e+04])
O2.name = 'O2'

O  = specie(15.9994, 0.0, 0.0, 59500, 1.5574e+07)
O.electro_g = np.array([5, 3, 1, 5, 1])
O.electro_th= np.array([0.0, 2.2770776e+02, 3.2656888e+02, 2.2830286e+04, 4.8619930e+04])
O.name = 'O'

N = specie(14.0067, 0, 0, 113200, 3.3747e+07)
N.electro_g = np.array([4, 10, 6])
N.electro_th= np.array([0.0, 2.7664696e+04, 4.1493093e+04])
N.name = 'N'

NO = specie(30.0061, 2817.0, 2.5, 75500, 3.0091e+6)
NO.electro_g = np.array([4, 8, 2, 4, 4, 4, 4, 2, 4, 2, 4, 4, 2, 2, 2, 4])
NO.electro_th= np.array([0.0, 5.4673458e+04, 6.3171396e+04, 6.5994503e+04, 6.9061210e+04, 7.0499985e+04,
                        7.4910550e+04, 7.6288753e+04, 8.6761885e+04, 8.7144312e+04, 8.8860771e+04,
                        8.9817556e+04, 8.9884459e+04, 9.0427021e+04, 9.0642838e+04, 9.1117633e+04])
NO.name = 'NO'

N2p= specie(28.0128514, 3395.0, 2.9, 113200, 5.3886e+07)
N2p.electro_g = np.array([2, 4, 2, 4, 8, 8, 4, 4, 4, 4, 
                         8, 8, 4, 4, 2, 2, 4])
N2p.electro_th= np.array([0.0, 1.3189972e+04, 3.6633231e+04, 3.6688768e+04, 5.9853048e+04, 6.6183659e+04, 
                          7.5989919e+04, 7.6255086e+04, 8.2010186e+04, 8.4168349e+04, 8.6326512e+04, 
                         8.9204062e+04, 9.2081613e+04, 9.2225490e+04, 9.2937684e+04, 9.6397938e+04, 1.0359181e+05])
N2p.name = 'N2p'

#O2p= specie(31.9982514, 2239, 2.1, 59500, 3.6592e+07)
#O2p.electro_g = np.array([3, 2, 1, 1, 6, 3, 3])
#O2p.electro_th = np.array([0.0, 1.1391560e+04, 1.8984739e+04, 4.7559736e+04, 4.9912421e+04, 5.0922686e+04, 7.1898633e+04])
#O2p.name = 'O2p'
O2p= specie(31.9982514, 2239, 2.1, 59500, 3.6592e+07)
O2p.electro_g = np.array([4, 8, 4, 6, 4, 2, 4, 4, 4, 4, 8, 4, 2, 2, 4])
O2p.electro_th = np.array([0.0, 4.7354408e+04, 5.8373987e+04, 5.8414273e+04, 6.2298966e+04, 6.7334679e+04,
                          7.1219372e+04, 7.6542841e+04, 8.8196920e+04, 8.8916307e+04, 9.4239776e+04, 9.4959163e+04,
                          9.5920265e+04, 9.9850999e+04, 1.0359181e+05])
O2p.name = 'O2p'


NOp = specie(30.0055514, 2817, 2.5, 75500, 3.3000e+07)
NOp.electro_g = np.array([1, 3, 6, 6, 3, 1, 2, 2])
NOp.electro_th= np.array([0.0, 7.5089678e+04, 8.5254624e+04, 8.9035726e+04, 9.7469826e+04, 1.0005530e+05, 1.0280337e+05, 1.0571386e+05])
NOp.name = 'NOp'

Op = specie(15.9988514, 0.0, 0.0, 59500, 9.8056e+07)
Op.electro_g = np.array([4, 10, 6])
Op.electro_th = np.array([0.0, 3.8583347e+04, 5.8223492e+04])
Op.name = 'Op'

Np = specie(14.0061514, 0.0, 0.0, 113200, 1.3438e+08)
Np.electro_g = np.array([1, 3, 5, 5, 1, 5, 15])
Np.electro_th = np.array([0.0, 7.0068352e+01, 1.8819180e+02, 2.2036569e+04, 4.7031835e+04, 6.7312522e+04, 1.3271908e+05])
Np.name = 'Np'

em = specie(0.00054858, 0.0, 0.0, 0.0)
em.electro_g = np.array([1.0])
em.electro_th = np.array([0.0])
em.name = 'em'

