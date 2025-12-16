import string
import torch
import numpy as np
from .BondIntegral import get_skf_tensors
from .Elements import symbol_to_number, label, atomic_num, mass
from .io import read_xyz
from .Tools import load_spinw_to_matrix, load_spinw_to_tensor

class Constants(torch.nn.Module):

    """
    Constants used in DFTB
    """

    def __init__(self, file, skfpath, magnetic_hubbard_ldep=False, param_grad=False):
        """
        Constructor
        """

        super().__init__()

        self.skfpath = skfpath
        self.symbol_to_number = symbol_to_number
        self.label = label
        self.atomic_num = atomic_num

        self.shell_dim = torch.nn.Parameter(torch.tensor([0,1,3,5], dtype=torch.int64),   requires_grad=False)
        self.atomic_num   = torch.nn.Parameter(atomic_num,   requires_grad=False)
        self.mass   = torch.nn.Parameter(mass,   requires_grad=False)

        if isinstance(file, str):
            species, _ = read_xyz([file], sort=False) #Input coordinate file
        else:
            species, _ = read_xyz(file, sort=False) #Input coordinate file
        TYPE = torch.tensor(species.flatten(), dtype=torch.int64)

        R_tensor, R_orb, coeffs_tensor, R_rep_tensor, rep_splines_tensor, \
        N_ORB, MAX_ANG, MAX_ANG_OCC, TORE, N_S, N_P, N_D, ES, EP, ED, US, UP, UD = get_skf_tensors(TYPE, self.skfpath)

        #w = load_spinw_to_tensor(skfpath + '/spinw.txt', device=TYPE.device)

        try:
            w_shell = load_spinw_to_matrix(skfpath + '/spinw.txt', device=TYPE.device)
            self.w_shell = torch.nn.Parameter(w_shell,   requires_grad=False)
        except:
            print("Warning: could not load spinw.txt file for spin-orbit coupling. Proceeding without SOC.")
            self.w_shell = None

        w_atom = torch.zeros(self.w_shell.shape[0], device=TYPE.device)
        w_atom[TYPE] = self.w_shell[TYPE, MAX_ANG_OCC[TYPE]-1, MAX_ANG_OCC[TYPE]-1]
        self.w_atom = torch.nn.Parameter(w_atom, requires_grad=False)

        if magnetic_hubbard_ldep:
            self.w = torch.nn.Parameter(w_shell.clone(), requires_grad=False)
        else:
            self.w = torch.nn.Parameter(w_atom.clone(), requires_grad=False)

        self.R_tensor = torch.nn.Parameter(R_tensor,   requires_grad=False)
        self.R_orb = torch.nn.Parameter(R_orb,   requires_grad=False)
        self.coeffs_tensor = torch.nn.Parameter(coeffs_tensor,   requires_grad=False)
        self.R_rep_tensor = torch.nn.Parameter(R_rep_tensor,   requires_grad=False)
        self.rep_splines_tensor = torch.nn.Parameter(rep_splines_tensor,   requires_grad=False)

        self.n_orb   = torch.nn.Parameter(N_ORB,   requires_grad=False)
        self.max_ang = torch.nn.Parameter(MAX_ANG, requires_grad=False) # AO with max angular momentum l
        self.max_ang_occ = torch.nn.Parameter(MAX_ANG_OCC, requires_grad=False) # occupied AO with max angular momentum l
        self.tore   = torch.nn.Parameter(TORE,   requires_grad=False)
        self.n_s = torch.nn.Parameter(N_S, requires_grad=False)
        self.n_p = torch.nn.Parameter(N_P, requires_grad=False)
        self.n_d = torch.nn.Parameter(N_D, requires_grad=False)

        self.U    = torch.nn.Parameter( US,   requires_grad=False)
        self.Up   = torch.nn.Parameter( UP,   requires_grad=False)
        self.Ud   = torch.nn.Parameter( UD,   requires_grad=False)
        self.Es   = torch.nn.Parameter( ES,   requires_grad=param_grad)
        self.Ep   = torch.nn.Parameter( EP,   requires_grad=param_grad)
        self.Ed   = torch.nn.Parameter( ED,   requires_grad=param_grad)

    def forward(self):
        pass


class ConstantsTest(torch.nn.Module):
    """
    Class for debugging.
    Constants used in DFTB.
    """

    def __init__(self):
        """
        Constructor
        """

        super().__init__()


        self.label=np.array(['0',
            'H',                                                                                            'He',
            'Li','Be',                                                            'B', 'C',  'N', 'O', 'F', 'Ne',
            'Na','Mg',                                                            'Al','Si',' P',' S', 'Cl','Ar',
            'K', 'Ca','Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga','Ge',' As','Se','Br','Kr',
            'Rb','Sr','Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In','Sn',' Sb','Te','I','Kr',
            'Cs','Ba','La', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl','Pb',' Bi','Po','At','Rn'])

        n_orb=torch.as_tensor([0,
            1,                                          1,
            1,1,                              4,4,4,4,4,4,
            1,1,                              9,9,9,9,9,9,
            1,1,9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,9,9,9,9,9,
            ],
            dtype=torch.int64)
        
        max_ang=torch.as_tensor([0,
            1,                                          1,
            1,1,                              2,2,2,2,2,2,
            1,1,                              3,3,3,3,3,3,
            1,1,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,3,3,3,3,3,
            ],
            dtype=torch.int64)

        tore=torch.as_tensor([0,
            1,                                             0,
            1,2,                                 3,4,5,6,7,0,
            1,2,                                 3,4,5,6,7,0,
            1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3,4,5,6,7,0,
            1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3,4,5,6,7,0,
            1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3,4,5,6,7,0],
            dtype=torch.int64 )
        
        # number of s electrons per atom
        n_s=torch.as_tensor([0,
            1,                                             2,
            1,2,                                 2,2,2,2,2,2,
            1,2,                                 2,2,2,2,2,2,
            1,2,2, 2, 2, 2, 2, 2, 2, 2, 2, 2,    2,2,2,2,2,2,
            1,2,2, 2, 2, 2, 2, 2, 2, 2, 2, 2,    2,2,2,2,2,2,
            1,2,2, 2, 2, 2, 2, 2, 2, 2, 2, 2,    2,2,2,2,2,2],
            dtype=torch.int64 )

        # number of p electrons per atom
        n_p=torch.as_tensor([0,
            0,                                             0,
            0,0,                                 1,2,3,4,5,6,
            0,0,                                 1,2,3,4,5,6,
            0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    1,2,3,4,5,6,
            0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    1,2,3,4,5,6,
            0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    1,2,3,4,5,6,],
            dtype=torch.int64)
        # number of d electrons per atom
        n_d=torch.as_tensor([0,
            0,                                             0,
            0,0,                                 0,0,0,0,0,0,
            0,0,                                 0,0,0,0,0,0,
            0,0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,   0,0,0,0,0,0,
            0,0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,   0,0,0,0,0,0],
            dtype=torch.int64)
        
        atomic_num=torch.as_tensor([0,
            1,                                                                  2,
            3,  4,                                          5,  6,  7,  8,  9,  0,
            11, 12,                                         13, 14, 15, 16, 17, 0,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0,],
            dtype=torch.int64)

        mass=torch.as_tensor([ 0.00000,
            1.00790,                                                                                                                                                                                   4.00260,
            6.94000,   9.01218,                                                                                                                10.81000,  12.01100,  14.00670,  15.99940,  18.99840,  20.17900,
        22.98977,  24.30500,                                                                                                                26.98154,  28.08550,  30.97376,  32.06000,  35.45300,  39.94800,
        39.09800,  40.07800,  44.95600,  47.86700,  50.94200,  51.99600,  54.93800,  55.84500,  58.93300,  58.69300,  63.54600,  65.38000,  69.72300,  72.63000,  74.92200,  78.97100,  79.90400,  83.79800,
        85.46800,  87.62000,  88.90600,  91.22400,  92.90600,  95.95000,  97.00000, 101.07000, 102.91000, 106.42000, 107.87000, 112.41000, 114.82000, 118.71000, 121.76000, 127.60000, 126.90000, 131.29000,
        132.91000, 137.33000, 174.97000, 178.49000, 180.95000, 183.84000, 186.21000, 190.23000, 192.22000, 195.08000, 196.97000, 200.59000, 204.38000, 207.20000, 208.98000, 209.00000, 210.00000, 222.00000 ],
            )



        # U_dict = torch.as_tensor([ 0.0,
        # 11.415182,                                                                                                                                                                                  0.0,
        #  0.0     ,  0.0     ,                                                                                                                0.0     ,  9.923997, 11.725390, 13.480530,  0.0     ,  0.0,
        #  0.0     ,  0.0     ,                                                                                                                0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
        #  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  6.297942,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
        # ],
        # )

        ####
        U_dict = torch.as_tensor([ 0.0,
        12.054683,                                                                                                                                                                                  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     ,  14.240811, 11.725390, 11.8761410,  0.0     ,  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
         0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  6.297942,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
        ],
        )

        Up_dict = torch.as_tensor([ 0.0,
         0.0     ,                                                                                                                                                                                  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     , 10.542376, 12.419827, 14.239861,  0.0     ,  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
         0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  5.146546,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
        ],
        )

        Ud_dict = torch.as_tensor([ 0.0,
         0.0     ,                                                                                                                                                                                  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
         0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     , 11.056617,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0     ,  0.0,
        ],
        )

        # Es_dict = torch.as_tensor([ 0.00000,
        # -6.492650686,                                                                                                                                                                                      0.0,
        # 0.0,          0.0,                                                                                                                   0.0, -13.73881005, -17.4153, -23.91426072,       0.0,       0.0,
        # 0.0,          0.0,                                                                                                                   0.0,  0.0        ,  0.0    ,  0.0        ,       0.0,       0.0,
        # 0.0,          0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0, -4.33922737,       0.0,       0.0,   0.0,  0.0        ,  0.0    ,  0.0        ,       0.0,       0.0,
        # ],
        # )
        
        ###
        Es_dict = torch.as_tensor([ 0.00000,
        -6.4835,                                                                                                                                                                                      0.0,
        0.0,          0.0,                                                                                                                   0.0, -13.7199, -17.4153, -23.9377,       0.0,       0.0,
        0.0,          0.0,                                                                                                                   0.0,  0.0        ,  0.0    ,  0.0        ,       0.0,       0.0,
        0.0,          0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0, -4.33922737,       0.0,       0.0,   0.0,  0.0        ,  0.0    ,  0.0        ,       0.0,       0.0,
        ],
        )

        # Ep_dict = torch.as_tensor([ 0.00000,
        # 0.0,                                                                                                                                                                                                0.0,
        # 0.0,       0.0,                                                                                                                     0.0,  -5.28867445,  -7.09477309,  -9.03776739,       0.0,       0.0,
        # 0.0,       0.0,                                                                                                                     0.0,   0.0       ,   0.0       ,   0.0       ,       0.0,       0.0,
        # 0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,   -0.75801842,       0.0,       0.0,   0.0,   0.0       ,   0.0       ,   0.0       ,       0.0,       0.0,
        # ],
        # )

        ###
        Ep_dict = torch.as_tensor([ 0.00000,
        0.0,                                                                                                                                                                                                0.0,
        0.0,       0.0,                                                                                                                     0.0,  -5.2541,  -7.09477309,  -9.0035,       0.0,       0.0,
        0.0,       0.0,                                                                                                                     0.0,   0.0       ,   0.0       ,   0.0       ,       0.0,       0.0,
        0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,   -0.75801842,       0.0,       0.0,   0.0,   0.0       ,   0.0       ,   0.0       ,       0.0,       0.0,
        ],
        )

        Ed_dict = torch.as_tensor([ 0.00000,
        0.0,                                                                                                                                                                                       0.0,
        0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
        0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
        0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,   -4.79872151,       0.0,       0.0,   0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
        ],
        )

        self.shell_dim = torch.nn.Parameter(torch.tensor([0,1,3,5], dtype=torch.int64),   requires_grad=False)

        self.n_orb   = torch.nn.Parameter(n_orb,   requires_grad=False)
        self.tore   = torch.nn.Parameter(tore,   requires_grad=False)
        self.atomic_num   = torch.nn.Parameter(atomic_num,   requires_grad=False)
        self.mass   = torch.nn.Parameter(mass,   requires_grad=False)
        self.U   = torch.nn.Parameter(U_dict,   requires_grad=False)
        self.Up   = torch.nn.Parameter(Up_dict,   requires_grad=False)
        self.Ud   = torch.nn.Parameter(Ud_dict,   requires_grad=False)
        self.Es   = torch.nn.Parameter(Es_dict,   requires_grad=False)
        self.Ep   = torch.nn.Parameter(Ep_dict,   requires_grad=False)
        self.Ed   = torch.nn.Parameter(Ed_dict,   requires_grad=False)
        self.max_ang = torch.nn.Parameter(max_ang, requires_grad=False)
        self.n_s = torch.nn.Parameter(n_s, requires_grad=False)
        self.n_p = torch.nn.Parameter(n_p, requires_grad=False)
        self.n_d = torch.nn.Parameter(n_d, requires_grad=False)

        self.skfpath = skfpath


    def forward(self):
        pass


