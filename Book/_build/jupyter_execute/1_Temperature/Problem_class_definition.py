#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'Reaction_class_definition.ipynb')


# # Problem class definition
# A problem class is defined so that several condition can be studied, analyzed, compared and contrasted.
# 
# The initial conditions for the post-shock relaxation region are obtained by assuming frozen chemistry through the shock which, in the Euler equations framework is a discontinuity whose jumps are defined by the Rankine-Hugoniot relations.
# 
# The variable are mixture density, velocity, temperature and mass fractions.
# 
# For the mixture model considered the the following relations hold:
# * $e_{mixture} = \Sigma_i e_i Y_i $
# * $CV_{mixture} = \Sigma_i CV_i Y_i $
# * $R_{mixture} = \Sigma_i R_i Y_i $
# 
# where Y is the mass fraction of each specie (obviously $ \Sigma_i Y_i = 1$ ).   

# ## Initialization

# In the following the class is created and the functions to compute the thermodynamic variables are defined.

# ```{note}
# Some variables have a default value when the class is initialized. All of them will be overwritten when defining a problem, but the sigma value that represents the equivalent cross section of pure nitrogen assuming hard sphere potential.
# ```

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
    
    def energy(self, Y, T):
        e, CV = 0, 0
        for x in range(len(self.specie)):
            e1, CV1 = self.specie[x].energy(T)
            e, CV = e + e1 * Y[x], CV + CV1*Y[x]
        return e, CV        
    
    def e_Y(self, Y, T):
        '''Computes the derivative of energy wrt Y at constant T'''
        e_Y = np.zeros(np.shape(self.Y0))
        for x in range(len(e_Y)):
            e_Y[x], inutile = self.specie[x].energy(T)
        return e_Y
        
    def only_e(self, Y , T):
        e, CV = 0, 0
        for x in range(len(self.specie)):
            if debug:
                print('only_e x : ' + str(x))
                print('only e Y : ' + str(Y))
            e1, CV1 = self.specie[x].energy(T)
            e = e + e1 * Y[x]
        return e    
    
    def T_from_e(self, Y, e, T0 = 1e3):
        
        T, infodict, ier, mesg = opt.fsolve(lambda T : self.only_e(Y, T) - e, x0 = T0, 
                                            xtol=1e-12, full_output=1)
        if not ier:
            print('T_from_e did not converge')
            
        return T


# ## Reynolds-Hugoiniot jump relations

# At first, the Rankine Hugoniot relations are solved for a perfect gas with the same R of the mixture and the isoentropic coefficient $\gamma = 1.4$. <br>
# Then, the results obtained are used as guess to initialize the non-linear solver. 
# 
# The Rankine-Hugoniot relations read: <br>
# 
# $ \rho_0 u_0 = \rho_1 u $ <br>
# $ \rho_0 u_0^2 + P_0 = \rho_1 u_1^2 + P_1 $ <br>
# $ h_0^t  = e_0 + \frac{P_0}{\rho_0} + \frac{1}{2}u_0^2 = 
# e_1 + \frac{P_1}{\rho_1} + \frac{1}{2} u_1^2 = h_1^t  $
# 
# The non-linear equations are written in the "RHsystem" function whose solutions are found through a non-linear solver. 

# ```{note}
# The shock is solved with the hypothesis of frozen chemistry.
# ```

# In[3]:


def RHsystem(self, x):
    rho2, T2, u2 = x
    p2 = rho2 * self.R(self.Y0) * T2
    e2, CV2 = self.energy(self.Y0, T2)
    
    p0 = self.rho0 * self.R(self.Y0) * self.T0
    e0, CV0 = self.energy(self.Y0, self.T0)
    
    out_rho = (self.rho0 * self.u0)                       - (rho2 * u2)
    out_mom = (self.rho0 * self.u0 ** 2 +  p0)            - (rho2 * u2**2 + p2)
    out_ene = (e0 + p0 / self.rho0 + 1 / 2 * self.u0 ** 2)- (e2 + p2 / rho2 + 1 / 2 * u2 ** 2 )
      
    out = [out_rho, out_mom, out_ene]
    
    if debug:
        print('Rho2, T2, u2 : ' + str(rho2), '/', str(T2), '/', str(u2))
        print(str(out))
        
    return out

    
def RHjump(self):
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
    x, infodict, ier, mesg = opt.fsolve(lambda x : self.RHsystem(x), x0 = [rho2, T2, u2], xtol=atol*1e-2, 
                                     full_output=1)

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
    print('Speed  : ' + str(self.u1))
    print('Mach   : ' + str(M2c))
    print('******************************')
    print('Reference mean free path : ' + str(self.mfp))
    print()
    if x[2] < 0:
        print('speed is negative!!! Look at RHjump')
        sys.exit("EXITING")

problem.RHsystem = RHsystem
problem.RHjump = RHjump


# ```{note}
# In the "RHjump" function a reference mean free path is computed to be able to compare the relaxation length in terms if mean free path for different initial conditions. It represents the mean free path just after the shock and is computed by the cross section defined in the problem class
# ```

# ```{warning}
# While the defualt value of sigma is reasonable for air (since the bi-atomical nitrogen is its the first specie in terms of composition both in mass and molar fractions), it is not exact. Furthermore the effects of different composition are not captured by this value, hence it shall be used with caution and only as a reference.
# ```

# ## Computation of the chemical source terms

# To compute the chemical sources for each specie $\omega_i$, for each reaction the specie mass fraction vector is transformed by an incidence matrix in "local" molar concentrations $\chi_i = \rho_i/M_i$ which are multiplied by the forward and backward reaction coefficients as defined before to compute the reaction rate. <br>
# According to the Park's definition of forward and backward coefficients, the reaction rate is computed as: <br>
# 
# $R_{f,r} = k_{f,r} \Pi_{i = 0}^{N_{s}} \chi_i^{\alpha_{i,r}} $ <br>
# and <br>
# $R_{b,r} = k_{b,r} \Pi_{i = 0}^{N_{s}} \chi_i^{\beta_{i,r}} $ <br>
# Where $\alpha_{i,r}$ and $\beta_{i,r}$ are the stechiometric mole numbers for reactants and products respectively.
# 
# <!-- Finally, the generated/destroyed species are expressed in terms of "global" mass fraction rate and summed for each reaction and sub-reaction. -->
# 
# <!-- [Aggiungere] and the released energy is computed -->
# 
# The net mass rate of production of the specie $i$ is obtained considering all the $N_r$ possible reactions, and it is defined as: <br>
# $
# \omega_i = M_i \sum_{r}^{N_r} \omega_{i,r} = M_i \sum_{r}^{N_r} \left(\beta_{i,r} - \alpha_{i,r} \right) \left( R_{f,r} - R_{b,r} \right)
# $
# 

# ```{note}
# Altough the energy released by the reaction is computed, it is no of practical use for the way the mixture energy has been written.
# ```

# In[4]:


def compute_Sy(self, rho, T, Y):
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
                
                # Transform the global y vector to the local n vector
                nr_l = np.matmul( rho * Y / mmol, np.transpose(omegar))
                np_l = np.matmul( rho * Y / mmol, np.transpose(omegap))
                
                # Compute the reaction rate
                w_s = obj.kf(T) * np.prod(nr_l ** obj.stoichr) - obj.kb(T) * np.prod(np_l ** obj.stoichp)
                #breakpoint()
                # Update the source terms for the species equation
                S += mmol * (np.matmul(obj.stoichp, omegap) - np.matmul( obj.stoichr, omegar)) * w_s
                #Update the energy source term
                Se = obj.e_mol * w_s
                
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

                # Transform the global y vector to the local n vector
                nr_l = np.matmul( rho * Y / mmol, np.transpose(omegar))
                np_l = np.matmul( rho * Y / mmol, np.transpose(omegap))
                
                # Compute the reaction rate
                w_s = obj.kf(T) * np.prod(nr_l ** obj.stoichr) - obj.kb(T) * np.prod(np_l ** obj.stoichp)
                
                # breakpoint()
                # Update the source terms for the species equation
                S += mmol * (np.matmul(obj.stoichp, omegap) - np.matmul( obj.stoichr, omegar)) * w_s
                #Update the energy source term
                Se = obj.e_mol * w_s
                
                
        else: print('Member of the reaction group of this problem are ill-defined')
    
    if debug:
        print('Se = : ' + str(Se))
    
    return S, Se

problem.compute_Sy = compute_Sy


# ## Pre-shock relax chemistry funciton

# In general, once the density and temperature of a mixture are known, the equilibrium composition can be computed by the chemical equilibrium constant. In this function, the equilibrium composition of the mixture is computed through the chemical source terms.

# In[5]:


def pre_shock_relax_chemistry(self, relax = 1e-2):
    output = 1
    Y  = self.Y0
    Sy = self.compute_Sy(self.rho0, self.T0, Y)
    
    while (any(abs(Sy) > 1e-8)):
        
        Sy = self.compute_Sy(self.rho0, self.T0, Y)
    
        Y += relax * Sy # Relaxation parameter to favour convergence of chemical relaxation for mixture very far from equilibrium
        if output :
            for i in range(len(self.specie)):
                print('********************************')
                print(self.specie[i].name + ' concentration : ' + str(Y[i]) + '    Sorgente : '  + str (Sy[i]))
                print('********************************')
            
        if not (np.sum(Y) - 1 < 1e-8):
            print('Species not conserved in pre-shock relax')
            
        if any(Y < 0):
            print('Y less than 0 in chemical relax pre - shock')
            
    return Y

problem.pre_shock_relax_chemistry = pre_shock_relax_chemistry


# ```{warning}
# Since in general both temperature, density and composition may be unknown or uncertain, this function prints the chemical equilibrium composition BUT DOES NOT UPDATE the inital conditions of the problem.
# ```

# ```{Note}
# This is not the most efficient way to compute chemical equilibrium composition, but given the code already written, it's the easiest to implement.
# ```

# ## Euler system of equation
# 
# The primary variables considered are density, velocity, mass fractions and temperature. <br>
# 
# The Euler equations read:
# 
# 
# * Mass equation: <br />
#     $ \frac{\partial \rho u}{\partial x} = \frac{\partial \rho}{\partial x}u + \rho \frac{\partial u}{\partial x} = 0$ 
#     
#     
# * Momentum equation: <br />
#     $ \rho u \frac{\partial u}{\partial x} = - \frac{\partial P}{\partial x} $ <br>
#     Since $ P = P(\rho, T, Y) = \rho \Sigma_i Y_i R T $ , then $ dp = \frac{\partial P}{\partial \rho} d \rho + \frac{\partial P}{\partial T} d T + \Sigma_i \frac{\partial P}{\partial Y_i} d Y_i $  <br>
#     The derivatives can be expressed as : <br>
#     - $ \frac{\partial P}{\partial \rho} = \Sigma_i Y_i R_i T $ <br>
#     - $ \frac{\partial P}{\partial T} = \rho \Sigma_i Y_i R_i$ <br>
#     - $ \frac{\partial P}{\partial Y_i} = \rho R_i T$ <br>
#     Hence, the momentum equation can be written as : <br>
#     $ \rho u \frac{\partial u}{\partial x} = - \frac{\partial P}{\partial x} = \Sigma_i Y_i R T \frac{\partial \rho}{\partial x} + \rho \Sigma_i Y_i R_u \frac{\partial T}{\partial x} + \rho R T \Sigma_i \frac{Y_i}{x}$    
#     
# * Energy equation: <br />
#     $ \frac{\partial e}{\partial x} = \frac{P}{\rho^2} \frac{\partial \rho}{\partial x}$ <br>
#     In analogy with the pressure, we have $ e = e (T, Y_i) $, then $ de = \frac{\partial e }{\partial T} dT + \Sigma \frac{\partial e }{\partial Y_i} dY_i$ <br>
#     The derivatives can be expressed as : <br>
#     - $ \frac{\partial e}{\partial T} = cv(T, Y_i) $ <br>
#     - $ \frac{\partial e}{\partial Y_i} = e_i(T) $ <br>
#     Hence, the energy equation can be written as : <br>
#     $ \frac{\partial T}{\partial x} = \frac{1}{cv} \left[ \frac{P}{\rho^2} \frac{\partial \rho}{\partial x} - \Sigma_i e_i(T) \frac{\partial Y_i}{\partial x} \right] $ 
#     
# * Species transport equation: <br />
#     $ \rho u \frac{\partial Y_i }{\partial x} = \omega_i \qquad for \; i = 1 ... N_s $

# Being the equations non linear, in analogy to what done for the shock relaxation, "Euler_system" defines the system of equations while "Euler_x" computes the derivative of the states by a non-linear solver.

# In[6]:


def Euler_system(self, x, x_x):
    rho, u, T = x[0], x[1], x[2]
    Y = x[3:]
    rho_x, u_x, T_x = x_x[0], x_x[1], x_x[2]
    Y_x = x_x[3:]
    
    p = self.R(Y) * rho * T 
    
    e2, cv = self.energy( Y, T )
    e_Y    = self.e_Y( Y, T)
    
    p_x = 0.0

    for t in range(len(self.specie)):
        p_x += rho_x * Y[t]  * Ru / self.specie[t].mmol * T +                rho * Y_x[t] * T * Ru / self.specie[t].mmol +                rho * Y[t] * Ru / self.specie[t].mmol * T_x

    if debug:
        print('First term of the derivative p_x: '+ str(rho_x * Y  * Ru / self.specie[1].mmol * T))
    
    S, Se = self.compute_Sy(rho, T, Y)
    
    if debug:
        print('Euler system R                                : ' + str(self.R(Y)))
        print('Euler system rho_x            : ' + str(rho_x))
        print('Euler system Y                : ' + str(Y))
        print('Euler system Y_x              : ' + str(Y_x))
        print('Euler system T                : ' + str(T))
        print('Euler system rho              : ' + str(rho))
        print('Euler system cv               : ' + str(cv))
        print('Euler system T_x              : ' + str(T_x))
        print('Euler system p                : ' + str(p))
        print('Euler system Se               : ' + str(Se))

    # Mass equation
    rho_xc = - rho / u * u_x
    
    # Momentum equation
    u_xc  = -  p_x / rho / u
    
    # Energy equation
    T_xc  = (p / rho ** 2 * rho_x - np.sum(e_Y * Y_x)) / cv
    
    # Species equations
    Y_xc  = S / rho / u
    
    x_xc = [rho_xc]
    x_xc.append(u_xc)
    x_xc.append(T_xc)
    
    for i in Y_xc:
        x_xc.append(i)
    
    
    if debug:
        print('Euler system x_x              : ' + str(x_x))
        print('Euler system x_c              : ' + str(x_xc))
        print('Euler system x_x - x_c        : ' + str(x_x - x_xc))
        print('Euler system S                : ' + str(S))
        print('Euler system Se               : ' + str(Se))
        
        
    return   x_x - x_xc
    
def Euler_x(self, x_spatial, x):
        
        print('A Solving for x = %.12e' %x_spatial, end="")
        print("\r", end="")
        
        if debug:
                print('State values  = ' + str(x))
        
        
        x0 = [0.0, 0.0, 0.0]
        
        for i in self.specie:
            x0.append(0)
        
        x0 = np.array(x0)
        
        x_x, infodict, ier, mesg = opt.fsolve(lambda x_x : self.Euler_system(x, x_x), 
                                            x0=x0,  
                                            xtol=atol*1e-2, full_output=1)
        if not ier:
            print('Euler_x did not converge')    
        
        if debug:
            print('ier = ' + str(ier))
            
        
        return x_x
    
    
problem.Euler_system = Euler_system
problem.Euler_x = Euler_x


# ## Solve

# The solve function resolves the shock and then computes the state varibles by integration of the derivatives in the spatial coordinate x.

# ```{note}
# The problem is stiff, thus an appropriate method must be chosen. In this code an implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation has been chosen. The first step is a fraction of the reference post-shock mean free path.
# ```

# In[7]:


def solve(self, xf = 1):
    
    # Compute post shock values to be used as initial conditions
    self.RHjump()
    
    y0 = [self.rho1, self.u1, self.T1]
    
    for i in self.Y0:
        y0.append(i)
    
    sol = itg.solve_ivp(self.Euler_x, 
                              [0.0, xf], y0, method='BDF', t_eval=None, dense_output=True, first_step=self.mfp / 1e4,
                              events=None, vectorized=False, args=None, rtol=rtol, atol=atol)
    
    self.sol_rho, self.sol_u, self.sol_T, self.sol_Y = sol.y[0,:], sol.y[1,:], sol.y[2,:], sol.y[3:,:]

    self.sol_x = sol.t
    self.sol = sol
    
problem.solve = solve


# ## Post-processing

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
        print(str(parameter))
        X[:,i] = X[:,i] / parameter
        
    self.sol_X = X
    self.sol_e = e
    self.sol_p = p
    
problem.postprocess = postprocess


# ## Plot

# Several plot functions have been prepared to ease analysis and comparisons. These function plot the outcome of the analysis with a double x-axis: the upper values represent the number of reference mean free path while the lower is the distance measured in meters from the shock.

# ## Temperature plot

# In[9]:


def plot_T(self, ax = None, xmax = None, xmax_l = None, ls='-'):
    # Set axes
    
    if ax == None:
        ax = plt.axes()
    
    # Set xmax
    
    if not xmax:
        if xmax_l:
            xmax = xmax_l * self.mfp
            
        else: xmax = self.sol_x[-1]
        
    x_lambda = self.sol_x / self.mfp
    
    ax.plot(self.sol_x, self.sol_T, ls)

    
    # Add second x axis 
    
    ax2 = ax.twiny()
    
    ax2.plot(x_lambda, self.sol_T, ls)

    
    # Set labels etc.
    ax.set_xlim(0, xmax)
    ax2.set_xlim(0, xmax / self.mfp)
    
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('x / mfp [-]')
    ax.set_ylabel('T [K]')
    ax2.grid()
    ax.yaxis.grid(True)
    
problem.plot_T = plot_T


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
    
problem.plot_rho = plot_rho


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
    
problem.plot_u = plot_u


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

problem.plot_Y = plot_Y


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
    
problem.plot_X = plot_X


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

problem.logplot_Y = logplot_Y


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

problem.logplot_X = logplot_X


# ## Validation

# For validation purposes, the fluxes of conservative variables are computed, the relative error is plotted and the max values printed. <br>
# The conservative fluxes are:
# * Mass flux : $ \rho u$
# * Momentum flux: $ \rho u^2 + P $
# * Energy flux: $ \left[ \rho \left( e + \frac{1}{2} u ^ 2 \right) + P \right] u $ 

# In[16]:


def validate(self, xmax = None, xmax_l = None, ls = '-', print_max = True):
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
    enthalpy_flux = (self.sol_rho * ( self.sol_e + 1 / 2 * self.sol_u ** 2 ) + self.sol_p) * self.sol_u 
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
    subaxes_mass_frac     = plt.subplot(4, 1, 5)
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
        print('Maximum enthalpy flux error   : ' + str(np.max(np.abs(error_enthalpy_flux))))
        print('Maximum mass frac error       : ' + str(np.max(np.abs(error_mass_frac_flux))))
        print('Last value of mass flux       : ' + str(mass_flux[-1]))
        print('Ymin                          : ' + str(np.min(self.sol_Y)))
        
problem.validate = validate

