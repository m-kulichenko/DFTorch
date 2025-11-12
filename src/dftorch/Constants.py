import torch
import numpy as np


class Constants(torch.nn.Module):

    """
    Constants used in DFTB
    """

    def __init__(self, skfpath):
        """
        Constructor
        """

        super().__init__()

        self.symbol_to_number = {
                        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
                        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
                        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
                        'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35
                        }


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
            4,1,                              9,9,9,9,9,9,
            1,1,9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,9,9,9,9,9,
            ],
            dtype=torch.int64)
        
        max_ang=torch.as_tensor([0,
            1,                                          1,
            1,1,                              2,2,2,2,2,2,
            2,1,                              3,3,3,3,3,3,
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


        U_dict = torch.as_tensor([ 0.0,
        11.415182,                                                                                                                                                                                  0.0,
         0.0     ,  0.0     ,                                                                                                                0.0     ,  9.923997, 11.725390, 13.480530,  0.0     ,  0.0,
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

        Es_dict = torch.as_tensor([ 0.00000,
        -6.492650686,                                                                                                                                                                                      0.0,
        0.0,          0.0,                                                                                                                   0.0, -13.73881005, -17.4153, -23.91426072,       0.0,       0.0,
        0.0,          0.0,                                                                                                                   0.0,  0.0        ,  0.0    ,  0.0        ,       0.0,       0.0,
        0.0,          0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0, -4.33922737,       0.0,       0.0,   0.0,  0.0        ,  0.0    ,  0.0        ,       0.0,       0.0,
        ],
        )
        
        Ep_dict = torch.as_tensor([ 0.00000,
        0.0,                                                                                                                                                                                                0.0,
        0.0,       0.0,                                                                                                                     0.0,  -5.28867445,  -7.09477309,  -9.03776739,       0.0,       0.0,
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

        self.U_dict_read = torch.zeros_like(U_dict)
        self.Up_dict_read = torch.zeros_like(Up_dict)
        self.Ud_dict_read = torch.zeros_like(Ud_dict)
        self.Es_dict_read = torch.zeros_like(Es_dict)
        self.Ep_dict_read = torch.zeros_like(Ep_dict)
        self.Ed_dict_read = torch.zeros_like(Ed_dict)

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

    def set_params(self, r_grad=False):
        self.U   = torch.nn.Parameter( self.U_dict_read.to(self.n_orb.device),    requires_grad=False)
        self.Up   = torch.nn.Parameter(self.Up_dict_read.to(self.n_orb.device),   requires_grad=False)
        self.Ud   = torch.nn.Parameter(self.Ud_dict_read.to(self.n_orb.device),   requires_grad=False)
        self.Es   = torch.nn.Parameter(self.Es_dict_read.to(self.n_orb.device),   requires_grad=r_grad)
        self.Ep   = torch.nn.Parameter(self.Ep_dict_read.to(self.n_orb.device),   requires_grad=r_grad)
        self.Ed   = torch.nn.Parameter(self.Ed_dict_read.to(self.n_orb.device),   requires_grad=r_grad)

    def forward(self):
        pass
class ConstantsTest(torch.nn.Module):

    """
    Constants used in DFTB
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


