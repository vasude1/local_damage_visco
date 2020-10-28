import numpy as np

class ViscousMaterial:
    def __init__(self,dashpots):
        self.E_inf = 1E6
        self.density = 1E3
        self.numberdashpots = dashpots
        self.g = np.zeros((self.numberdashpots,1))
        self.tau = np.zeros((self.numberdashpots,1))
        self.E_o = 0.0
        self.Y_o = 0.0
        self.Y_c = 1E-10
        self.E = 0.0
        # 2E-12
        return

class InternalVariables:
    def __init__(self,num_dashpots,element_length):
        self.numberdashpots = num_dashpots
        self.H = np.zeros((self.numberdashpots,1))
        self.damage = 0.0
        self.old_disp = np.zeros((2,1)) # at nodes
        self.sigma = 0.0
        self.epsilon = 0.0
        self.SE = 0.0
        self.le = element_length
        return
