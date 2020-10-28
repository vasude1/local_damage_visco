import numpy as np

## Computes Strain in an element/ at a gp
def compute_epsilon(disp,le):
    return((disp[1]-disp[0])/le)

## Computes H variables in an element/ at a gp
def compute_H(H_old,new_disp,old_disp,tau,delta_t,le):
    H_new = np.zeros((len(H_old),1))
    for i in range(len(H_old)):
        H_new[i] += np.exp(-delta_t/tau[i])*H_old[i]
        H_new[i] += np.exp(-delta_t/2.0/tau[i])*(compute_epsilon(new_disp,le)-compute_epsilon(old_disp,le))
    return H_new

## Computes Strain Energy in an element/ at a gp
def compute_strainenergy(new_disp,le,H_array,damage,material):
    eps = compute_epsilon(new_disp,le)
    psi = 0.0
    psi += 0.5*material.E_inf*eps**2
    for i in range(material.numberdashpots):
        psi += 0.5*material.E_o*material.g[i]*H_array[i]**2
    return (1-damage)*psi

## Computes damage in an element
def compute_damage(old_damage,strain_energy,Y_o,Y_c,delta_t):
    # k=0.01
    # if(strain_energy < Y_c):
    #     return old_damage
    # old_damage += delta_t*k*(strain_energy/Y_c - 1.0)
    # if(old_damage >1.0):
    #     old_damage = 0.99
    ######### Warning
    # print(old_damage)
    new_damage = (np.sqrt(strain_energy)-np.sqrt(Y_o))/(np.sqrt(Y_c)-np.sqrt(Y_o))
    if(new_damage > 1.0):
        new_damage = 1.0
    if(old_damage>new_damage):
        new_damage = old_damage
    return new_damage

## Uniform body force
def body_force(t):
    return 0.001*t

## Computes the stiffness (scalar) for whole body
def compute_effectivestiff(material,delta_t):
    E = 0.0
    E += material.E_inf
    for i in range(material.numberdashpots):
        E += np.exp(-delta_t/2.0/material.tau[i][0])*material.E_o*material.g[i][0]
    return E

def compute_Mass(number_elements,le,density):
    Mass = np.zeros((number_elements+1,number_elements+1))
    ele_array = np.array(([2,1],[1,2]))
    for i in range(number_elements):
        Mass[i:i+2,i:i+2] += 1.0/6.0*density*le*ele_array
    return Mass

def compute_Stress(IV,material,delta_t):
    stress = 0.0
    stress += material.E_inf*IV.epsilon
    old_str = compute_epsilon(IV.old_disp,IV.le)
    for i in range(IV.numberdashpots):
        stress += material.E_o*material.g[i]*IV.H[i]
        # *np.exp(-delta_t/material.tau[i])
        # stress -= material.E_o*material.g[i]*np.exp(-delta_t/2.0/material.tau[i])*old_str
    return (1-IV.damage)*stress
