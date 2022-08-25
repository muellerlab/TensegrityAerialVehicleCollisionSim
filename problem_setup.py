"""
With the same mass budget for tensegrity and propeller guard,
compare the max stress in the system during the process of an external hit.
"""

class design_param():
    def __init__(self):
        self.mStructure = 50/1000.0 #[kg] mass of the protection structure
        self.mQuad = 250/1000.0 #[kg] mass of the quadcopter 

        propInch = 2.5 # 2.5in propeller diameter
        self.propR = propInch*(25.4)/(2*1000) #[m], radius of propeller

        # rod
        self.rRho = 2000 #[kg/m^3] density of carbon fiber 
        self.rE = 32e9 #[Pa] Young's modulus of carbon fiber 
        self.rUS = 3.5e9 #[Pa] Ultimate strenth of carbon fiber "https://dragonplate.com/what-is-carbon-fiber"

        # string
        self.sRho = 1150; # [kg/m^3] density of nylon string
        self.sE = 4.1e9 #[Pa] Young's modulus of string 
        self.sD = 1e-3  #[m] diameter of string 
        self.gamma_m = 20 # ratio between mass of rod and string 
        self.sPreT = 20  # [N] string pre-tension force

        """
        Two different ways to design vehicles: 

        0: given propeller size, assume that the propeller placement is symmetrical, find shortest rod length that can enclose it.
        Meanwhile, the prop guard protects the same vehicle put on tensegrtiy

        1: given propeller size, find the shortest tensegrity that can hold it without the symmetrical assumption. 
        Meanwhile, the prop guard is the smallest possible design to frame the propellers so they don't ht each other. 
        """
        self.designCase = 1

        self.dString = 500 #[N/(m/s)] 
        self.dRod = 500 #[N/(m/s)]
        self.dJoint = 10 #[Nm/(rad/s)] Rotational damping constant
        return
