import torch
import numpy as np
import re

def write_XYZ_trajectory(filename, structure, comment, step=0, Ftot=None):
	with open(filename, 'a+') as f:
		num_atoms = structure.Nats
		f.write(f"{structure.Nats}\n")
		f.write(f"Step {step}, {comment}\n")
		for i in range(num_atoms):
			x = structure.RX[i].item()
			y = structure.RY[i].item()
			z = structure.RZ[i].item()
			symbol = structure.TYPE[i].item()
			if Ftot is not None:
				fx = Ftot[i, 0].item()
				fy = Ftot[i, 1].item()
				fz = Ftot[i, 2].item()
				f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")
			else:
				f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
                    
def write_XYZ(filename, structure, comment, step=0, Ftot=None):
	with open(filename, 'a+') as f:
		num_atoms = structure.Nats
		f.write(f"{structure.Nats}\n")
		f.write(f"Step {step}, {comment}\n")
		for i in range(num_atoms):
			x = structure.RX[i].item()
			y = structure.RY[i].item()
			z = structure.RZ[i].item()
			symbol = structure.TYPE[i].item()
			if Ftot is not None:
				fx = Ftot[i, 0].item()
				fy = Ftot[i, 1].item()
				fz = Ftot[i, 2].item()
				f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")
			else:
				f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

def write_xyz_from_xyz(filename, TYPE, RX, RY, RZ, comment, step=0, Ftot=None):
	with open(filename, 'a+') as f:
		num_atoms = len(TYPE)
		f.write(f"{num_atoms}\n")
		f.write(f"Step {step}, {comment}\n")
		for i in range(num_atoms):
			x = RX[i].item()
			y = RY[i].item()
			z = RZ[i].item()
			symbol = TYPE[i].item()
			if Ftot is not None:
				fx = Ftot[i, 0].item()
				fy = Ftot[i, 1].item()
				fz = Ftot[i, 2].item()
				f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")
			else:
				f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def read_xyz(files, sort = True):
    '''
    reads xyz structure from a list (files) of files names
    '''
    element_dict = {
                'H':  1,                                                                                                                                 'He':2,
                'Li': 3, 'Be': 4,                                                                                'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne':10,
                'Na': 11,'Mg':12,                                                                                'Al':13,'Si':14,'P': 15,'S': 16,'Cl':17,'Ar':18,
                'K':  19,'Ca':20,'Sc':21,'Ti':22,'V': 23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,
                'Rb': 37,'Sr':38,'Y': 39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I': 53,'Xe':54,
                }

    COORDINATES = []
    for file in files:
        f = open(file)
        lines = f.readlines()
        f.close()
        Natoms = int(lines[0])
        coords = []
        try:
            int(lines[2].split()[0])
            atoms_are_number = True
        except:
            atoms_are_number = False
        for i in range(2, 2+Natoms):
            #species.append(int(lines[i].split()[0]))
            if atoms_are_number:
                coords.append([int(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2]), float(lines[i].split()[3])])
            else:
                coords.append([element_dict[lines[i].split()[0]], float(lines[i].split()[1]), float(lines[i].split()[2]), float(lines[i].split()[3])])
        COORDINATES.append(coords)
    COORDINATES = np.array(COORDINATES)
    if sort: COORDINATES = np.array([x[(-1*x[ :, 0]).argsort()] for x in COORDINATES])

    SPECIES =COORDINATES[:,:,0].astype(int)
    COORDINATES = COORDINATES[:,:,1:4]

    return SPECIES, COORDINATES

def read_xyz_traj_data(file):
    '''
    Reads data from xyz trajectory comment lines at each step
    e.g. "Step 0, Etot = -145495.406250 eV, Epot = -145702.171875 eV, Ekin = 206.764328 eV, T = 399.90 K, Res = 0.000000, mu = -2.5353 eV"
    Returns a dict with 'step' (int array) and any parsed named quantities as numpy arrays.
    Missing values are filled with NaN.
    '''

    steps = []
    data = {}  # key -> list of values (aligned with steps)
    with open(file, 'r') as fh:
        lines = fh.readlines()

    i = 0
    nlines = len(lines)
    while i < nlines:
        line = lines[i].strip()
        if line == '':
            i += 1
            continue
        # expect number-of-atoms line
        try:
            natoms = int(line)
        except ValueError:
            # skip malformed line
            i += 1
            continue
        # comment line should be next
        if i + 1 >= nlines:
            break
        comment = lines[i + 1].strip()

        # extract step number if present
        m = re.search(r'\bStep\b\s*([0-9]+)', comment, re.IGNORECASE)
        step = int(m.group(1)) if m else None
        steps.append(step)

        # extract key=value pairs (value is a float, may have units after)
        found = {}
        for key, val in re.findall(r'([A-Za-z_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*[A-Za-z%°K]*)?', comment):
            try:
                found[key] = float(val)
            except Exception:
                found[key] = np.nan

        # update existing keys with either found value or NaN
        for k in list(data.keys()):
            data[k].append(found.get(k, np.nan))
        # add any newly seen keys, backfilling previous steps with NaN
        for k, v in found.items():
            if k not in data:
                data[k] = [np.nan] * (len(steps) - 1)  # previous steps
                data[k].append(v)
        # advance to next frame
        i += natoms + 2

    result = {'step': np.array(steps, dtype=float)}
    for k, lst in data.items():
        result[k] = np.array(lst, dtype=float)

    return result