#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', '_2T_Reaction_class_definition.ipynb')


# # Problem class definition
# In the same manner of the one temperature, the problem class is a container for mixture thermodynamics, initial conditions, solution and plot functions. 

# ## Thermodynamics
# On one hand many thermodynamic functions are the same as their one temperature equivalent, on the other hand it is necessary to define appropriate functions for the computation of the translational-rotational energy form the translational-rotational temperature and the same for the vibro-electronic.

# In[2]:


class problem:
    def __init__(self):
        self.specie = []
        self.Y0     = []
        self.T0     = 300
        self.rho0   = 1.225
        self.u0     = 3000
        self.reaction = []
        self.sigma     = 3.14 * (6.2 * 1e-10) ** 2 # Sigma HS nitrogen
    
    def add_specie_ic(self, specie, Y0):
        
        if not self.specie.count(specie):
            self.specie.append(specie)
            self.Y0.append(Y0)
        
        else:
            idx = self.specie.index(specie)
            self.Y0[idx] = Y0
        
    def R(self, Y):
        R = 0
        for x in range(len(self.specie)):
            R += self.specie[x].R * Y[x]
        return R
    
    def energy_2T(self, Y, T, Tv):
        e_tr, e_ve, cv_tr, cv_ve = 0, 0, 0, 0
        for x in range(len(self.specie)):
            e1_tr, e1_ve, cv1_tr, cv1_ve = self.specie[x].energy_2T(T, Tv)
            e_tr  = e_tr + e1_tr * Y[x]
            cv_tr = cv_tr + cv1_tr * Y[x]
            e_ve  = e_ve + e1_ve * Y[x]
            cv_ve = cv_ve + cv1_ve *Y [x]
        return e_tr, e_ve, cv_tr, cv_ve
    
    def e_Y_2T(self, Y, T, Tv):
        '''Computes the derivative of energy 2T wrt Y at constant T'''
        e_Y_tr = np.zeros(np.shape(self.Y0))
        e_Y_ve = np.zeros(np.shape(self.Y0))
        for x in range(len(e_Y_tr)):
            e_Y_tr[x], e_Y_ve[x], useless_tr, useless_ve = self.specie[x].energy_2T(T, Tv)
        return e_Y_tr, e_Y_ve
    
    def e_Y_v(self, Y, T, Tv):
        '''Computes the derivative of vibrational-electronic energy wrt Y at constant T'''
        e_Y_v = np.zeros(np.shape(self.Y0))
        for x in range(len(e_Y_v)):
            e_Y_v[x] = self.specie[x].energy_vib_2T(Tv)
        return e_Y_v
    
    def only_e_tr(self, Y, T):
        '''Computes the translational-rotational energy'''
        e_tr = 0
        for x in range(len(self.specie)):
            if debug:
                print('only_e x : ' + str(x))
                print('only e Y : ' + str(Y))
            e1_tr, e1_ve, cv1_tr, cv1_ve = self.specie[x].energy_2T(T, T)
            e_tr = e_tr + e1_tr * Y[x]
        return e_tr
    
    def only_e_ve(self, Y, Tv):
        '''Computes the vibrational-electronic energy'''
        e_ve = 0
        for x in range(len(self.specie)):
            if debug:
                print('only_e x : ' + str(x))
                print('only e Y : ' + str(Y))
            e1_ve = self.specie[x].energy_vib_2T(Tv)
            e_ve = e_ve + e1_ve * Y[x]
        return e_ve
    
    def T_from_e_2T(self, Y, e_tr, e_ve, T0_tr = 1e3, T0_ve = 1e3):
        
        T_tr, infodict, ier_tr, mesg = opt.fsolve(lambda T : self.only_e_tr(Y, T) - e_tr, x0 = T0_tr,
                                            xtol=atol * 1e-2, full_output=1)
        T_ve, infodict, ier_ve, mesg = opt.fsolve(lambda Tv : self.only_e_ve(Y, Tv) - e_ve, x0 = T0_ve,
                                            xtol=atol * 1e-2, full_output=1)
        if not ier_tr:
            print('T_tr_from_e did not converge')
        if not ier_ve:
            print('T_ve_from_e did not converge')
            
        return T_tr, T_ve
    


# ## RH jump relations

# The Rankine-Hugoniot relations are the same and read: <br>
# 
# 
# $ \rho_0 u_0 = \rho_1 u_1 $ <br>
# $ \rho_0 u_0^2 + P_0 = \rho_1 u_1^2 + P_1 $ <br>
# $ h_0^t  = e_0 + \frac{P_0}{\rho_0} + \frac{1}{2}u_0^2 = 
# e_1 + \frac{P_1}{\rho_1} + \frac{1}{2} u_1^2 = h_1^t $ <br>
# $ Y_{i_0} = Y_{i_1} $ <br>
# 
# 
# Where $P = \rho \sum_{i}^{N_s} Y_i R_i T $ <br>
# The non-linear equations are written in the "RHsystem" function whose solutions are found through a non-linear solver. 

# ```{note}
# The shock is solved with the hypothesis of frozen chemistry and constant vibrational-electronic temperature.
# ```

# In[3]:


def RHsystem_2T(self, x):
    rho2, T2, u2 = x
    p2 = rho2 * self.R(self.Y0) * T2
    e2_tr, e2_ve, CV2_tr, CV2_ve = self.energy_2T(self.Y0, T2, self.T0)
    
    p0 = self.rho0 * self.R(self.Y0) * self.T0
    e0_tr, e0_ve, CV0_tr, CV0_ve = self.energy_2T(self.Y0, self.T0, self.T0)
    
    out_rho = (self.rho0 * self.u0)                                  - (rho2 * u2)
    out_mom = (self.rho0 * self.u0 ** 2 +  p0)                       - (rho2 * u2**2 + p2)
    out_ene = (e0_tr + e0_ve + p0 / self.rho0 + 1 / 2 * self.u0 ** 2)- (e2_tr + e2_ve + p2 / rho2 + 1 / 2 * u2 ** 2 )
      
    out = [out_rho, out_mom, out_ene]
    
    if debug:
        print('Rho2, T2, u2 : ' + str(rho2), '/', str(T2), '/', str(u2))
        print(str(out))
        
    return out

    
def RHjump_2T(self):
    # Build a guess assuming perfect gas with gamma = 1.4
    gamma = 1.4
    c0 = (gamma * self.R(self.Y0) * self.T0) ** (1/2)
    p0 = self.rho0 * self.R(self.Y0) * self.T0
    M0 = self.u0 / c0
    M2 = (((gamma - 1) * M0 ** 2 + 2) / ( 2 * gamma * M0 ** 2 - ( gamma - 1))) ** ( 1 / 2); 
    rho2 = self.rho0 * ( gamma + 1 ) * M0 ** 2 / ( ( gamma - 1 ) * M0 ** 2 + 2); 
    p2 = p0 * ( 2 * gamma * M0 ** 2 - ( gamma - 1 ) ) / ( gamma + 1); 
    T2 = self.T0 * ( ( 2 * gamma * M0 ** 2 - ( gamma - 1 ) ) * ( ( gamma - 1 ) * M0 ** 2 + 2 ) )             / ( ( gamma + 1 ) ** 2 * M0 ** 2 ); 
    c2 = (gamma * self.R(self.Y0) * T2)**(1/2);
    u2 = c2 * M2;
        
    # Solve RH relations
    x, infodict, ier, mesg = opt.fsolve(lambda x : self.RHsystem_2T(x), x0 = [rho2, T2, u2], xtol=atol*1e-2, 
                                     full_output=1) # , epsfcn=1e-8, factor = 0.1

    if not ier:
        print('RH not converged')
            
    self.rho1, self.T1, self.u1 = x[0], x[1], x[2]
    
    # Compute reference post shock mean free path
    mmol = []
    
    for i in self.specie:
        mmol.append(i.mmol)
    
    mmixt = np.sum(np.array(mmol) * self.Y0)
    
    self.mfp = Kb * T2 / p2  / self.sigma
    
    e2, cv2 = self.energy(self.Y0, self.T1)
    cp = cv2 + self.R(self.Y0)
    gamma = cp / cv2
    c2c = (gamma * self.R(self.Y0) * self.T1) ** (1/2)
    M2c = self.u1 / c2c
    
    print('Pre shock Mach : ' + str(M0))
    print('******************************')
    print('Post-shock guess values:')
    print('rho    : ' + str(rho2))
    print('T      : ' + str(T2))
    print('Speed  : ' + str(u2))
    print('Mach   : ' + str(M2))
    print('******************************')
    print('Post-shock values:')
    print('rho    : ' + str(self.rho1))
    print('T      : ' + str(self.T1))
    print('Tv     : ' + str(self.T0))
    print('Speed  : ' + str(self.u1))
    print('Mach   : ' + str(M2c))
    print('******************************')
    print('Reference mean free path : ' + str(self.mfp))
    print()
    if x[2] < 0:
        print('speed is negative!!! Look at RHjump')
        sys.exit("EXITING")

problem.RHsystem_2T = RHsystem_2T
problem.RHjump_2T = RHjump_2T


# ## Computation of the chemical source terms

# The chemical source terms are computed in the same manner as the one temperaturee model but the forward and backward reaction rates depend both on the translational-rotational and vibrational-electronic temperature according to Park's model [1].

# In[4]:


def compute_Sy_2T(self, rho, T, Y, Tv):
    S = np.zeros(np.shape(self.specie))
    Y = np.array(Y)
    Se = 0
    # Recover mmol
    mmol = np.array([])
    for i in self.specie:
        mmol = np.append(mmol, i.mmol)
    
    
    for i in range(len(self.reaction)):
        if isinstance(self.reaction[i], reaction):
            for j in range(len(self.reaction[i].subreactions)):
                obj = self.reaction[i].subreactions[j]
                # Add the reactants to the reactants matrix by an incidence matrix omegar
                omegar = np.zeros([len(obj.reactants), len(self.specie)])
                for k in range(len(obj.reactants)):
                    idx = self.specie.index(obj.reactants[k])
                    # omegar[k][idx] = 1
                    omegar[k, idx] = 1
                
                # Add the products to the products matrix by an incidence matrix omegap
                omegap = np.zeros([len(obj.products), len(self.specie)])
                
                for k in range(len(obj.products)):
                    idx = self.specie.index(obj.products[k])
                    # omegap[k][idx] = 1
                    omegap[k, idx] = 1
                
                # Transform the global y vector to the local X vector
                chi_r_l = np.matmul( rho * Y / mmol, np.transpose(omegar))
                chi_p_l = np.matmul( rho * Y / mmol, np.transpose(omegap))
                
                # Compute the reaction rate
                R_s = obj.kf_2T(T, Tv) * np.prod(chi_r_l ** obj.stoichr) - obj.kb_2T(T, Tv) * np.prod(chi_p_l ** obj.stoichp)
                #breakpoint()
                # Update the source terms for the species equation
                S += mmol * (np.matmul(obj.stoichp, omegap) - np.matmul( obj.stoichr, omegar)) * R_s
                #Update the energy source term
                Se = obj.e_mol * R_s
                # S += mmol * (np.matmul(np.transpose(omegap), obj.stoichp) - np.matmul(  np.transpose(omegar), obj.stoichr)) * w_s
                
        elif isinstance(self.reaction[i], subreaction):
                obj = self.reaction[i]
                
                # Add the reactants to the reactants matrix by an incidence matrix omegar
                omegar = np.zeros([len(obj.reactants), len(self.specie)])
                for k in range(len(obj.reactants)):
                    idx = self.specie.index(obj.reactants[k])
                    # omegar[k][idx] = 1
                    omegar[k, idx] = 1
                
                # Add the products to the products matrix by an incidence matrix omegap
                omegap = np.zeros([len(obj.products), len(self.specie)])
                
                for k in range(len(obj.products)):
                    idx = self.specie.index(obj.products[k])
                    # omegap[k][idx] = 1
                    omegap[k, idx] = 1

                # Transform the global y vector to the local X vector
                chi_r_l = np.matmul( rho * Y / mmol, np.transpose(omegar))
                chi_p_l = np.matmul( rho * Y / mmol, np.transpose(omegap))
                
                # Compute the reaction rate
                R_s = obj.kf_2T(T, Tv) * np.prod(chi_r_l ** obj.stoichr) - obj.kb_2T(T, Tv) * np.prod(chi_p_l ** obj.stoichp)
                
                # breakpoint()
                # Update the source terms for the species equation
                S += mmol * (np.matmul(obj.stoichp, omegap) - np.matmul( obj.stoichr, omegar)) * R_s
                #Update the energy source term
                Se = obj.e_mol * R_s
                
                
        else: print('Member of the reaction group of this problem are ill-defined')
    
    if debug:
        print('Se = : ' + str(Se))
    
    return S, Se


problem.compute_Sy_2T = compute_Sy_2T


# ```{note}
# Now the reaction rates $R_{i,j}$ depend on the type of reaction considered as described above 
# ```

# ## Computation of the Vibrational-Translational Energy transfer and Chemical-Vibrational Coupling
# 
# The source term $S_v$ is taken into account in order to study the evolution of the vibrational-elecronic energy term introducted by the two-temperature model.<br>
# $ S_v = S_{c-v} + S_{v-t} $ <br>
# Where:
# * $\mathbf{S_{c-v}}$ is the vibrational energy lost or gained due to the chemical reactions. <br>
# $ S_{c-v} = \sum_{i}^{N_s} \omega_i e^{ve}_i $ <br>
# 
# * $\mathbf{S_{v-t}} $ is the vibrational energy relaxation term between vibrational and translational energy modes due to collisions.
# The energy exchange rate is modeled using the Landau-Teller model as: <br>
# $ S_{v-t} = \sum_i \rho_i \frac{e^{ve,*}_i - e^{ve}_i}{\tau_i} $ <br>
# Where $e^{ve,*}_i$ is the vibrational-electronic energy of species $i$ at the translational-rotational temperature $T^{tr}$ and $\tau_i$ is the characteristic relaxation time. For a mixture: <br>
# $ \tau_i = \frac{\sum_r X_r}{\sum_r X_r/\tau_{ir}} $ <br>
# $ X_i = \frac{Y_i}{M_i} \left( \sum_{r}\frac{Y_r}{M_r} \right)^{-1} $ <br>
# $\tau_{ir}$ is obtained from the Millikan and White's semiempirical correlation: <br>
# $ \tau_{ir} = \tau_{ir}^{MW} = \frac{101325}{p} exp \left[ A_{ir} \left(\left(T^{tr}\right)^{-1/3} - B_{ir} \right) - 18.42 \right] $ <br>
# Where:
# $ A_{ir} = 1.16 \cdot 10^{-3} \mu_{ir}^{1/2} \theta_{v,i}^{4/3} $ <br>
# $ B_{ir} = 0.015 \mu_{ir}^{1/4} $ <br>
# $ \mu_{ir} = \frac{M_i M_r}{M_i + M_r} $ <br>

# In[5]:


def compute_Sv(self, rho, T, Y, Tv, Sy):
    Y = np.array(Y)
    
    # Recover mmol
    mmol = np.zeros(len(self.specie))
    # Recover theta_v
    th_v = np.zeros(len(self.specie))
    for i in range(len(self.specie)):
        mmol[i] = self.specie[i].mmol
        th_v[i] = self.specie[i].th_v
    
    # Recover Pressure
    p = self.R(Y) * rho * T 
    
    ts = np.zeros(len(self.specie))
    e_v_eq = np.zeros(len(self.specie))
    e_v = np.zeros(len(self.specie))
    Svt = np.zeros(len(self.specie))
    X = Y * mmol
    X = X / np.sum(X)
    
    # Compute v-e and chemical source terms
    for j in range(len(self.specie)):
        # xr = Y[j] / mmol[j] / (np.sum(Y/mmol))
        xr = X
        mu_sr = mmol[j] * mmol / (mmol[j] + mmol)     # reduced molecular weight of colliding species
        A_sr = 1.16e-3 * mu_sr**(1/2) * th_v**(4/3) 
        B_sr = 0.015 * mu_sr**(1/4)
        tau_sr = 101325 / p * np.exp(A_sr * (T**(-1/3) - B_sr) - 18.42)
        ts[j] = np.sum(xr) / np.sum(xr / tau_sr)     
        e_v[j] = self.specie[j].energy_vib_2T(Tv)
        e_v_eq[j] = self.specie[j].energy_vib_2T(T)
        Svt[j] = rho * Y[j] * (e_v_eq[j] - e_v[j]) / ts[j]  
        
        
    Sc = Sy * e_v
    Sv = np.sum(Svt) + np.sum(Sc)
    
    if debug:
        print('Sv = : ' + str(Sv))
    
    return Sv

problem.compute_Sv = compute_Sv


# ## 2 Temperature version (Temperature version)

# ### Euler system of equation, 2 temperature version
# 
# The equations read:
# * $\textbf{Mass equation}$: <br />
#     $ \frac{\partial \rho u}{\partial x} = \frac{\partial \rho}{\partial x}u + \rho \frac{\partial u}{\partial x} = 0 \Longrightarrow \frac{\partial \rho}{\partial x} = - \frac{\rho}{u} \frac{\partial u}{\partial x} $
#     
# * $\textbf{Momentum equation}$: <br />
#     $ \rho u \frac{\partial u}{\partial x} = - \frac{\partial P}{\partial x} $ <br>
#     Since $ P = P(\rho, T^{tr}, Y) = \rho \Sigma_i Y_i R_i T^{tr} $ , then $ dp = \frac{\partial P}{\partial \rho} d \rho + \frac{\partial P}{\partial T^{tr}} d T^{tr} + \Sigma_i \frac{\partial P}{\partial Y_i} d Y_i $  <br>
#     The derivatives can be expressed as : <br>
#     - $ \frac{\partial P}{\partial \rho} = \Sigma_i Y_i R_i T^{tr} $ <br>
#     - $ \frac{\partial P}{\partial T^{tr}} = \rho \Sigma_i Y_i R_i$ <br>
#     - $ \frac{\partial P}{\partial Y_i} = \rho R_i T^{tr}$ <br>
#     Hence, the momentum equation can be written as : <br>
#     $ \rho u \frac{\partial u}{\partial x} = - \frac{\partial P}{\partial x} = \Sigma_i Y_i R_i T^{tr} \frac{\partial \rho}{\partial x} + \rho \Sigma_i Y_i R_i \frac{\partial T^{tr}}{\partial x} + \rho R T^{tr} \Sigma_i \frac{\partial Y_i}{\partial x}$    
#     

# ```{note}
# The momentum equation can be rewritten also as:
# $\frac{\partial u}{\partial x} =\frac{- \sum_i \frac{\partial p}{\partial Y_i} \frac{\partial Y_i}{\partial x} + \frac{\partial p}{\partial T} \left( \sum_i \frac{\partial e}{\partial Y_i} \frac{\partial Y_i}{\partial x} + \frac{\partial e}{\partial T^v} \frac{\partial T^v}{\partial x} \right) \left( \frac{\partial e}{\partial T} \right)^{-1}}{\rho u - \frac{\rho}{u} \frac{\partial P}{\partial \rho} - \frac{p}{\rho u} \frac{\partial P}{\partial T} \left( \frac{\partial e}{\partial T} \right)^{-1}} $
# ```

# * $\textbf{Energy equation}$: <br />
#     $ \frac{\partial e}{\partial x} = \frac{P}{\rho^2} \frac{\partial \rho}{\partial x}$ <br>
#     Expressing the dependence of e on the thermodynamic variables: $ e = e (T, T^v, Y_i) $, then <br>
#     $ de = \frac{\partial e }{\partial T} dT + \frac{\partial e }{\partial T^v} dT^v + \Sigma \frac{\partial e }{\partial Y_i} dY_i$ <br>
#     
#     The derivatives can be expressed as : <br>
#     - $ \frac{\partial e}{\partial T} = cv^{tr} (T, Y_i) $ <br>
#     - $ \frac{\partial e}{\partial T^v} = \frac{\partial e^v}{\partial T^v} = cv^v (T^v, Y_i) $ <br>
#     - $ \frac{\partial e}{\partial Y_i} = e_i(T) $ <br>
#     - $ \frac{\partial e^v}{\partial Y_i} = e^v_i(T^v) $ <br>
#     Hence, the energy equation can be written as : <br>
#     $ \frac{\partial T}{\partial x} = \left [ \frac{P}{\rho^2} \frac{\partial \rho}{\partial x} - \frac{\partial e}{\partial T^v} \frac{\partial T^v}{\partial x} - \Sigma_i \frac{\partial e}{\partial Y_i} \frac{\partial Y_i}{\partial x} \right ] \left( \frac{\partial e}{\partial T} \right)^{-1}$ <br>
#     
# * $\textbf{Vibrational energy equation}$: <br>
#     $ \rho u \frac{\partial e^v}{\partial x} = S_v $ <br>
#     Expressing the dependence of $e^v = e^v (T^v, Y_i)$ on the thermodynamic variables: 
#     $ de^v = \frac{\partial e^v }{\partial T^v} dT^v + \Sigma \frac{\partial e^v }{\partial Y_i} dY_i$ <br>
#     $ \frac{\partial T^v}{\partial x} = \left [ \frac{S_v}{\rho u} - \Sigma_i \frac{\partial e^v}{\partial Y_i} \frac{\partial Y_i}{\partial x} \right ] \left( \frac{\partial e^v}{\partial T^v} \right)^{-1}$
#     
# * $\textbf{Species transport equation}$: <br />
#     $ \rho u \frac{\partial Y_i }{\partial x} = \omega_i \qquad for \; i = 1 ... N_s $

# In[6]:


def Euler_system_2T(self, x, x_x):
    rho, u, T, Tv = x[0], x[1], x[2], x[3]
    Y = x[4:]
    rho_x, u_x, T_x, Tv_x = x_x[0], x_x[1], x_x[2], x_x[3]
    Y_x = x_x[4:]    
    
    p_x = 0
    mmol = np.array([])
    for t in range(len(self.specie)):
        mmol = np.append(mmol, self.specie[t].mmol)
        p_x += rho_x * Y[t]  * Ru / self.specie[t].mmol * T +                rho * Y_x[t] * T * Ru / self.specie[t].mmol +                rho * Y[t] * Ru / self.specie[t].mmol * T_x
    
    e2_tr, e2_ve, cv_tr, cv_ve = self.energy_2T( Y, T, Tv )
    e_Y_tr, e_Y_ve             = self.e_Y_2T( Y, T, Tv)
    e_Y_v                      = self.e_Y_v(Y, T, Tv)
    e_Y_tot = e_Y_tr + e_Y_ve 
    e_tr_T = cv_tr
    e_ve_Tv = cv_ve
    p_Y = rho * T * Ru / mmol
    p_T = np.sum(rho * Y * Ru / mmol)
    p_rho = np.sum(T * Y * Ru / mmol)
    
        
    if debug:
        print('First term of the derivative p_x: '+ str(rho_x * Y  * Ru / self.specie[1].mmol * T))
    
    p = self.R(Y) * rho * T  

    S, Se = self.compute_Sy_2T(rho, T, Y, Tv)
    Svt = self.compute_Sv(rho, T, Y, Tv, S)
    
    if debug:
        print('Euler system R                                : ' + str(self.R(Y)))
        print('Euler system rho_x            : ' + str(rho_x))
        print('Euler system Y                : ' + str(Y))
        print('Euler system Y_x              : ' + str(Y_x))
        print('Euler system T                : ' + str(T))
        print('Euler system Tv                : ' + str(Tv))
        print('Euler system rho              : ' + str(rho))
        print('Euler system cv               : ' + str(cv))
        print('Euler system T_x              : ' + str(T_x))
        print('Euler system Tv_x              : ' + str(Tv_x))
        print('Euler system p                : ' + str(p))
        print('Euler system Se               : ' + str(Se))
        print('Euler system Svt               : ' + str(Svt))

    # Mass equation
    rho_xc = - rho / u * u_x
    
    # Momentum equation
    # u_xc  = -  p_x / rho / u
    u_xc  = (- np.inner(p_Y, Y_x) + p_T / e_tr_T * (np.inner(e_Y_tot, Y_x) + e_ve_Tv * Tv_x) )/             (rho * u - rho / u * p_rho - p / (rho * u) * p_T / e_tr_T )
    
    # Energy equation                                           # new term : de_ve/dTv * dTv/dx
    T_xc  = (p / (rho ** 2) * rho_x - np.sum(e_Y_tot * Y_x) - e_ve_Tv * Tv_x) / e_tr_T
    Tv_xc = (Svt / (rho * u) - np.sum(e_Y_v * Y_x)) / e_ve_Tv
    
    # Species equations
    Y_xc  = S / rho / u
    
    x_xc = [rho_xc]
    x_xc.append(u_xc)
    x_xc.append(T_xc)
    x_xc.append(Tv_xc)
    
    for i in Y_xc:
        x_xc.append(i)
    
    
    if debug:
        print('Euler system x_x              : ' + str(x_x))
        print('Euler system x_c              : ' + str(x_xc))
        print('Euler system x_x - x_c        : ' + str(x_x - x_xc))
        print('Euler system S                : ' + str(S))
        print('Euler system Se               : ' + str(Se))
        print('Euler system Svt              : ' + str(Svt))
        
        
    return   x_x - x_xc
    
def Euler_x_2T(self, x_spatial, x, x0 = None):
        
        print('Solving for x = %.12e' %x_spatial, end="")
        print("\r", end="")
        
        
        if debug:
                print('State values  = ' + str(x))
        
        if x0 == None:
            a = 0.2
            x0 = [self.rho1, self.u1, self.T1, ( a * self.T1 + (1-a) * self.T0)]
            for i in self.Y0:
                x0.append(i)
        
        x0 = np.array(x0)
        
        x_x, infodict, ier, mesg = opt.fsolve(lambda x_x : self.Euler_system_2T(x, x_x), 
                                            x0=x0, xtol=atol*1e-2, full_output=1)
        if not ier:
            print('Euler_x did not converge')    
        
        if debug:
            # print('x_x = ' + str(x_x))
            print('ier = ' + str(ier))
            
            
        return x_x
    
    
problem.Euler_system_2T = Euler_system_2T
problem.Euler_x_2T = Euler_x_2T


# ### Solve, 2 temperature version

# The solve function is analogous to the one temperature version.

# In[7]:


def solve_2T(self, xf = 1):
    
    # Compute post shock values to be used as initial conditions
    self.RHjump_2T()
    
    y0 = [self.rho1, self.u1, self.T1, self.T0]
    
    for i in self.Y0:
        y0.append(i)
                                                                                    # True
    sol = itg.solve_ivp(self.Euler_x_2T, 
                              [0.0, xf], y0, method='BDF', t_eval=None, dense_output=True, first_step=self.mfp / 1e5,
                              events=None, vectorized=False, rtol=rtol, atol=atol)
    
    self.sol_rho, self.sol_u, self.sol_T, self.sol_Tv, self.sol_Y = sol.y[0,:], sol.y[1,:], sol.y[2,:], sol.y[3,:], sol.y[4:,:]

    self.sol_x = sol.t
    self.sol = sol
    
    # Compute energy
    e_tr = np.zeros(np.shape(self.sol_T))
    e_ve = np.zeros(np.shape(self.sol_T))
    p = np.zeros(np.shape(self.sol_T))
    
    for i in range(len(self.sol_T)):
        e_tr[i] = self.only_e_tr(self.sol_Y[:,i], self.sol_T[i])
        e_ve[i] = self.only_e_ve(self.sol_Y[:,i], self.sol_Tv[i])
        p[i] = self.sol_rho[i] * self.R(self.sol_Y[:,i]) * self.sol_T[i]
        
    self.sol_e_tr = e_tr
    self.sol_e_ve = e_ve
    self.sol_p = p
    
problem.solve_2T = solve_2T


# ## Post-processing
# 
# Once the result in terms of state variables have been computed, the pressure, energy and molar mass fractions are recovered to be used for the visualization and analysis of the results.

# In[8]:


def postprocess(self):
    # Post processing
    # Gather mmol of each specie
    mmol = np.array([])
    for i in self.specie:
        mmol = np.append(mmol, i.mmol)
        
    e = np.zeros(np.shape(self.sol_T))
    p = np.zeros(np.shape(self.sol_T))
    X = np.zeros(np.shape(self.sol_Y))
    for i in range(len(self.sol_T)):
        e[i] = self.only_e(self.sol_Y[:,i], self.sol_T[i])
        p[i] = self.sol_rho[i] * self.R(self.sol_Y[:,i]) * self.sol_T[i]
        X[:,i] = self.sol_Y[:,i] / mmol
        # Normalization of the molar fraction
        parameter = np.sum(X[:,i])
        # print(str(parameter))
        X[:,i] = X[:,i] / parameter
        
    self.sol_X = X
    self.sol_e = e
    self.sol_p = p
    
problem.postprocess = postprocess


# ## Plot

# Several plot functions have been prepared to ease analysis and comparisons. These function plot the outcome of the analysis with a double x-axis: the upper values represent the number of reference mean free path while the lower is the distance measured in meters from the shock.

# ### Temperature plot

# In[9]:


def plot_2T(self, ax = None, xmax = None, xmax_l = None, ls='-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.plot(self.sol_x, self.sol_T, ls, label = 'Trt')
    ax.plot(self.sol_x, self.sol_Tv, ls, label = 'Tv')
    ax.legend()

    # Add second x axis 
    
    ax2 = ax.twiny()
    
    ax2.plot(x_lambda, self.sol_T, ls)
    ax2.plot(x_lambda, self.sol_Tv, ls)

    # Set labels etc.
    #ax.set_ylim(bottom = 0)
    ax.set_xlim(0, xmax)
    ax2.set_xlim(0, xmax / self.mfp)
    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / mfp [-]')
    ax.set_ylabel('Tv [K]')
    ax2.grid()
    ax.yaxis.grid(True)


# ### Density plot

# In[10]:


def plot_rho(self, ax = None, xmax = None, xmax_l = None, ls='-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.plot(self.sol_x, self.sol_rho, ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
    
    ax2.plot(x_lambda, self.sol_rho, ls)

    
    # Set labels etc.
    ax.set_xlim(0, xmax)
    ax2.set_xlim(0, xmax / self.mfp)
    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / mfp [-]')
    ax.set_ylabel('rho [Kg/m3]')
    
    ax2.grid()
    ax.yaxis.grid(True)


# ### Velocity plot

# In[11]:


def plot_u(self, ax = None, xmax = None, xmax_l = None, ls='-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.plot(self.sol_x, self.sol_u, ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
    
    ax2.plot(x_lambda, self.sol_u, ls)

    
    # Set labels etc.
    ax.set_xlim(0, xmax)
    ax2.set_xlim(0, xmax / self.mfp)
    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / mfp [-]')
    ax.set_ylabel('Velocity [m/s]')
    
    ax2.grid()
    ax.yaxis.grid(True)


# ### Mass fractions plot

# In[12]:


def plot_Y(self, ax = None, xmax = None, xmax_l = None, ls = '-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.plot(self.sol_x, np.transpose(self.sol_Y), ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
        
    ax2.plot(x_lambda, np.transpose(self.sol_Y), ls)


    # Set labels etc.
    ax.set_xlim(0, xmax)
    #ax.set_ylim(bottom = 0)
    ax2.set_xlim(0, xmax / self.mfp)

    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / reference mfp [-]')
    ax.set_ylabel('Mass fractions')
    
    label = []
    for i in self.specie:
        label.append(i.name)
        
    ax.legend(label)
    
    ax2.grid()
    ax.yaxis.grid(True)


# ### Molar fractions plot

# In[13]:


def plot_X(self, ax = None, xmax = None, xmax_l = None, ls = '-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.plot(self.sol_x, np.transpose(self.sol_X), ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
        
    ax2.plot(x_lambda, np.transpose(self.sol_X), ls)


    # Set labels etc.
    ax.set_xlim(0, xmax)
    #ax.set_ylim(bottom = 0)
    ax2.set_xlim(0, xmax / self.mfp)

    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / reference mfp [-]')
    ax.set_ylabel('Molar fractions')
    
    label = []
    for i in self.specie:
        label.append(i.name)
        
    ax.legend(label)
    
    ax2.grid()
    ax.yaxis.grid(True)


# ### Mass fractions log-plot

# In[14]:


def logplot_Y(self, ax = None, xmax = None, xmax_l = None, ls = '-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.semilogy(self.sol_x, np.transpose(self.sol_Y), ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
        
    ax2.semilogy(x_lambda, np.transpose(self.sol_Y), ls)


    # Set labels etc.
    ax.set_xlim(0, xmax)
    #ax.set_ylim(bottom = 0)
    ax2.set_xlim(0, xmax / self.mfp)
    ax.set_ylim(1e-5,1)
    ax2.set_ylim(1e-5,1)

    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / reference mfp [-]')
    ax.set_ylabel('Mass fractions')
    
    label = []
    for i in self.specie:
        label.append(i.name)
        
    ax.legend(label)
    
    ax2.grid()
    ax.yaxis.grid(True)


# ### Molar fractions log-plot

# In[15]:


def logplot_X(self, ax = None, xmax = None, xmax_l = None, ls = '-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.semilogy(self.sol_x, np.transpose(self.sol_X), ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
        
    ax2.semilogy(x_lambda, np.transpose(self.sol_X), ls)


    # Set labels etc.
    ax.set_xlim(0, xmax)
    #ax.set_ylim(bottom = 0)
    ax2.set_xlim(0, xmax / self.mfp)
    ax.set_ylim(1e-5,1)
    ax2.set_ylim(1e-5,1)

    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / reference mfp [-]')
    ax.set_ylabel('Molar fractions')
    
    label = []
    for i in self.specie:
        label.append(i.name)
        
    ax.legend(label)
    
    ax2.grid()
    ax.yaxis.grid(True)


# ## Validation
# 
# For validation purposes, the fluxes of conservative variables are computed, the relative error is plotted and the max values printed. <br>
# The conservative fluxes are:
# * Mass flux : $ \rho u$
# * Momentum flux: $ \rho u^2 + P $
# * Energy flux: $ ( \rho ( e + \frac{1}{2} u ^ 2 ) + P ) u $ 

# In[16]:


def validate_2T(self, xmax = None, xmax_l = None, ls = '-', print_max = True):
    '''For validation purpuses, the fluxes of conserved values are plotted as a function of x'''
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    # Compute and plot mass flux

    subaxes_mass       = plt.subplot(4, 1, 1)
    mass_flux = self.sol_rho * self.sol_u
    error_mass_flux = ( mass_flux - mass_flux[0] ) / mass_flux[0]
    subaxes_mass.plot(self.sol_x, error_mass_flux, ls)
    
    subaxes_mass2 = subaxes_mass.twiny()
    subaxes_mass2.plot(x_lambda, error_mass_flux, ls)
    
    # Set labels etc.
    subaxes_mass.set_xlim(0, xmax)
    subaxes_mass2.set_xlim(0, xmax / self.mfp)
    subaxes_mass.set_xlabel('x [m]')
    subaxes_mass2.set_xlabel('x / reference mfp [-]')
    subaxes_mass.set_ylabel('Mass flux [-]')
    subaxes_mass2.grid()
    subaxes_mass.yaxis.grid(True)
    
    # Compute and plot momentum flux
    subaxes_momentum   = plt.subplot(4, 1, 2)
    momentum_flux = self.sol_rho * self.sol_u ** 2 + self.sol_p 
    error_momentum_flux = ( momentum_flux - momentum_flux[0] ) / momentum_flux[0]
    subaxes_momentum.plot(self.sol_x, error_momentum_flux, ls)
    
    subaxes_momentum2 = subaxes_momentum.twiny()
    subaxes_momentum2.plot(x_lambda, error_momentum_flux, ls)
    
    # Set labels etc.
    subaxes_momentum.set_xlim(0, xmax)
    subaxes_momentum2.set_xlim(0, xmax / self.mfp)
    subaxes_momentum.set_xlabel('x [m]')
    subaxes_momentum2.set_xlabel('x / reference mfp [-]')
    subaxes_momentum.set_ylabel('Momentum flux [-]') 
    subaxes_momentum2.grid()
    subaxes_momentum.yaxis.grid(True)
    
    # Compute and plot total enthalpy flux
    subaxes_enthalpy     = plt.subplot(4, 1, 3)
    enthalpy_flux = (self.sol_rho * ( self.sol_e_tr + self.sol_e_ve + 1 / 2 * self.sol_u ** 2 ) + self.sol_p) * self.sol_u 
    error_enthalpy_flux = ( enthalpy_flux - enthalpy_flux[0] ) / enthalpy_flux[0]
    subaxes_enthalpy.plot(self.sol_x, error_enthalpy_flux, ls)
    
    subaxes_enthalpy2 = subaxes_enthalpy.twiny()
    subaxes_enthalpy2.plot(x_lambda, error_enthalpy_flux, ls)
    
    # Set labels etc.
    subaxes_enthalpy.set_xlim(0, xmax)
    subaxes_enthalpy2.set_xlim(0, xmax / self.mfp)
    subaxes_enthalpy.set_xlabel('x [m]')
    subaxes_enthalpy2.set_xlabel('x / reference mfp [-]')
    subaxes_enthalpy.set_ylabel('enthalpy flux [-]') 
    subaxes_enthalpy2.grid()
    subaxes_enthalpy.yaxis.grid(True)
    
    # Compute and plot sum of mass fractions
    subaxes_mass_frac     = plt.subplot(4, 1, 4)
    mass_frac_flux = np.sum(self.sol_Y,axis=0) 
    error_mass_frac_flux = ( mass_frac_flux - mass_frac_flux[0] ) / mass_frac_flux[0]
    subaxes_mass_frac.plot(self.sol_x, error_mass_frac_flux, ls)
    
    subaxes_mass_frac2 = subaxes_mass_frac.twiny()
    subaxes_mass_frac2.plot(x_lambda, error_mass_frac_flux, ls)
    
    # Set labels etc.
    subaxes_mass_frac.set_xlim(0, xmax)
    subaxes_mass_frac2.set_xlim(0, xmax / self.mfp)
    subaxes_mass_frac.set_xlabel('x [m]')
    subaxes_mass_frac2.set_xlabel('x / reference mfp [-]')
    subaxes_mass_frac.set_ylabel('Sum of mass fractions [-]')
    subaxes_mass_frac2.grid()
    subaxes_mass_frac.yaxis.grid(True)
    
    if print_max:
        print('Maximum mass flux error       : ' + str(np.max(np.abs(error_mass_flux))))
        print('Maximum momentum flux error   : ' + str(np.max(np.abs(error_momentum_flux))))
        print('Maximum energy flux error     : ' + str(np.max(np.abs(error_enthalpy_flux))))
        print('Maximum mass frac error       : ' + str(np.max(np.abs(error_mass_frac_flux))))
        print('Last value of mass flux       : ' + str(mass_flux[-1]))
        print('Ymin                          : ' + str(np.min(self.sol_Y)))


# In[17]:


problem.plot_2T = plot_2T
problem.plot_rho = plot_rho
problem.plot_u = plot_u
problem.plot_Y = plot_Y
problem.plot_X = plot_X
problem.logplot_Y = logplot_Y
problem.logplot_X = logplot_X
problem.validate_2T = validate_2T 

