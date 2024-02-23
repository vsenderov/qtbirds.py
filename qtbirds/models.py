from enum import Enum
import numpy as np

class PhenotypicModel(Enum):
    MK_2 = 'mk-2state'
    MK_11 = 'mk-11state'
    NEIGHBOR_11 = 'neighbor-11state'

    def get_filename(self):
        return f"pheno/{self.value}.Q.json"
    
    def get_q_matrix(self):
        if self == PhenotypicModel.MK_2:
            return np.array([[-1., 1.],
                             [1., -1.]])
        elif self == PhenotypicModel.MK_11:
            # Placeholder for MK_11 Q-matrix
            # You will need to define this based on your specific requirements
            pass
        elif self == PhenotypicModel.NEIGHBOR_11:
            # Placeholder for NEIGHBOR_11 Q-matrix
            # You will need to define this based on your specific requirements
            pass
        else:
            raise NotImplementedError(f"Q-matrix for {self} is not implemented.")
    

class MolecularModel(Enum):
    HKY85 = "hky85"
    JC69 = "jc69"

    def get_filename(self):
        return f"nucleo/{self.value}.Q.json"
    
    def get_q_matrix(self):
        if self == MolecularModel.JC69:
            return np.array([[-1., 1/3, 1/3, 1/3],
                             [1/3, -1., 1/3, 1/3],
                             [1/3, 1/3, -1., 1/3],
                             [1/3, 1/3, 1/3, -1.]])
        elif self == MolecularModel.HKY85:
            # Placeholder for HKY85 Q-matrix
            # You will need to define this based on your specific requirements
            pass
        else:
            raise NotImplementedError(f"Q-matrix for {self} is not implemented.")
