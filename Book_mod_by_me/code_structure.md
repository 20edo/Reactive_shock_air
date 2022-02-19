# Code structure
The main classes of the code are:
* Specie -> Contains specie data such as energy level and degeneracy and allows to compute the energy of a specie at a given temperature
* Reaction and sub-reaction -> Contain the reaction data such as reactants, products, exponents etc. and allows to compute the forward and backward coefficients
* Problem -> Contains the problem information such as initial mixture composition, speed, temperature, etc. and allows to compute the mixture properties for a given state, solve the chemical relaxation problem and plot the results.
