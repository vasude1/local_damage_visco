## Imports from Python
import numpy as np
import scipy.sparse as sps


## From own mods
from classes import *
from compute_Matrices import *


0.0761
## Material Parameters
dashpots = 8
prop = np.array([[0.9,1E-7],[0.01,1e-06],[0.00100388,1e-05],
                [0.00109131,0.0001],[0.00115114,0.001],[0.00121,0.01],
                [0.0061,0.1],[0.07,1]])
# prop=np.zeros((dashpots,2))

material = ViscousMaterial(dashpots)
for i in range(dashpots):
    material.g[i] = prop[i][0]
    material.tau[i] = prop[i][1]
material.E_o = material.E_inf/(1-sum(material.g[:][0]))

## Parameters for temporal loop
end_time = 1000000.0
delta_t = 0.5

## Parameters for geometry and mesh
length_of_bar = 1.0
number_elements = 25   # element has nodes (element,element+1)
element_length = length_of_bar/number_elements
nodes = element_length*np.arange(number_elements+1)


## Time integration
beta = 0.2
gamma = 0.6

## FE quantities
Mass = compute_Mass(number_elements,element_length,material.density)
# Mass = sps.coo_matrix(Mass)
Stiff = np.zeros((number_elements+1,number_elements+1))
f = np.zeros((number_elements+1,1))
bf = np.zeros((number_elements+1,1))
IV = [InternalVariables(dashpots,element_length) for _ in range(number_elements)]
u = np.zeros((number_elements+1,1))
delta_u = np.zeros((number_elements+1,1))
v = np.zeros((number_elements+1,1))
a = np.zeros((number_elements+1,1))
ele_array = 1.0/element_length*np.array(([1.0,-1.0],[-1.0,1.0]))
forc_array = element_length*np.array(([[0.5],[0.5]]))

conv_count = 0
time = 0.0
while(time<end_time):
    print(time,delta_t)
    time_old = time
    converge = bool(0)
    u_tmp = u[:]
    a_tmp = a[:]
    iter = 0
    # [1:] [1:,1:]
    while((not converge) and (iter<60)):
        material.E = compute_effectivestiff(material,delta_t)
        compute_Matrices(time_old+delta_t,number_elements,IV,material,delta_t,Stiff,bf,f,u_tmp)
        LHS = 1.0*(1.0/beta/delta_t**2 * Mass + Stiff)
        # print(np.dot(Mass,a_tmp),bf)
        # exit(0)
        RHS = -np.dot(Mass,a_tmp) - f + bf
        delta_u[1:] = np.linalg.solve(LHS[1:,1:],RHS[1:])
        u_tmp = u_tmp+delta_u
        a_tmp = 1.0/beta*((u_tmp-u)/delta_t**2 - v/delta_t - (0.5-beta)*a)
        if(np.linalg.norm(delta_u) < 1E-8):
            converge = bool(1)
        iter += 1

    if(not converge):
        delta_t = 0.5*delta_t
        conv_count = 0

    elif(converge):
        time = time_old + delta_t
        v = v+delta_t*((1-gamma)*a+gamma/beta*((u_tmp-u)/delta_t**2)-v/delta_t-(0.5-beta)*a)
        u[:] = u_tmp[:]
        a[:] = a_tmp[:]
        IV = update_internalvariables(time,number_elements,IV,material,delta_t,u)
        write_solution(u,a,IV,time_old)
        conv_count += 1
        if(conv_count > 2):
            delta_t = 1.5*delta_t
            conv_count = 0
