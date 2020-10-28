import numpy as np
import matplotlib.pyplot as plt

from required_functions import *
from classes import *

def compute_Matrices(time,number_elements,IV,material,delta_t,Stiff,bf,f,u):
    element_length = IV[0].le
    bf_scalar = body_force(time)
    ele_array = 1.0/element_length*np.array([[1.0,-1.0],[-1.0,1.0]])
    forc_array = element_length*np.array([[0.5],[0.5]])
    temp_IV = InternalVariables(IV[0].numberdashpots,element_length)
    for i in range(number_elements):
        temp_IV.H = compute_H(IV[i].H,u[i:i+2],IV[i].old_disp,material.tau,delta_t,temp_IV.le)
        temp_IV.damage = compute_damage(IV[i].damage,IV[i].SE,material.Y_o,material.Y_c,delta_t)
        temp_IV.SE = compute_strainenergy(u[i:i+2],IV[i].le,temp_IV.H,temp_IV.damage,material)
        temp_IV.epsilon = compute_epsilon(u[i:i+2],element_length)
        temp_IV.sigma = compute_Stress(temp_IV,material,delta_t)

        Stiff[i:i+2,i:i+2] += (1-temp_IV.damage)*material.E*ele_array
        f[i:i+2] += (1-temp_IV.damage)*temp_IV.sigma*forc_array
        bf[i:i+2] += bf_scalar*forc_array
        # material.density*


def update_internalvariables(time,number_elements,IV,material,delta_t,u):
    le = IV[0].le
    for i in range(number_elements):
        IV[i].H = compute_H(IV[i].H,u[i:i+2],IV[i].old_disp,material.tau,delta_t,le)
        IV[i].SE = compute_strainenergy(u[i:i+2],le,IV[i].H,IV[i].damage,material)
        IV[i].damage = compute_damage(IV[i].damage,IV[i].SE,material.Y_o,material.Y_c,delta_t)
        IV[i].epsilon = compute_epsilon(u[i:i+2],le)
        IV[i].sigma = compute_Stress(IV[i],material,delta_t)
        IV[i].old_disp[0] = u[i][0]
        IV[i].old_disp[1] = u[i+1][0]
    return IV


def write_solution(u,a,IV,time_old):
    # loc = np.zeros((len(IV),IV[0].numberdashpots))
    # for i in range(len(IV)):
    #     for j in range(IV[0].numberdashpots):
    #         loc[i,j] = IV[i].H[j]
    loc = np.zeros((len(IV),1))
    for i in range(len(IV)):
        loc[i] = IV[i].damage
    x_coord = IV[0].le*np.arange(len(IV))
    # plt.plot(loc)
    # plt.show()
    np.savetxt('coarse/damage'+str(time_old)+'.txt', np.c_[x_coord,loc])
    np.savetxt('IV.txt', np.c_[IV[0].damage])
