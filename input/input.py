'''
Author:      Kartikay Shukla
File:        input.py
Created:     October 28, 2026
LM:          October 28, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
'''

import json
import numpy as np 
from pathlib import Path

NEL = 4
NNPEL = 3
L = 100 # in
d = 1 # in
D = 2 # in
A1 = L * (D-d)
A2 = L * (D-d)
A3 = np.pi * (D**2 - d**2) / 4
I1 = np.pi * (D*84 - d**4) / 64
I2 = np.pi * (D*84 - d**4) / 64
J = np.pi * (D*84 - d**4) / 32
E = 30E6
nu = 0.25
G = E / 2 / (1+nu)

inputCond = {
        'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 25,
        'convergence': 1E-3,
        'loadsteps': 10,
        }

inputGeom = {'L': L,   # length of beam, in
        'd': d, # inner diameter of beam, in
        'D': D, # outer diameter of beam, in
        'A1': A1, # area of beam in in^2, of plane E1
        'A2': A2, # area of beam in in^2, of plane E2
        'A3': A3, # cross-sectional area of beam in in^2, of plane E3
        'I1': I1, # moment of inertia in^4, pi(D^4 - d^4)/64, inertia about E1xE1
        'I2': I2, # moment of inertia in^4, pi(D^4 - d^4)/64, inertia about E2xE2
        'J': J, # polar moment of inertia in^4, inertia about E3xE3, J=I1+I2
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0], [6]], # [node number], [no of boundary values at node]    
        'Values': [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }

inputNBC = {'globalNode#': [[16], [0]], 
            'Values': [[1, 0]] 
        }

inputForce = {'globalNode#': [[0], [0]], 
            'Values': [[1, 0]] 
        }

inputMoment = {'globalNode#': [[16], [0]], 
            'Values': [[1, 0]] 
        }

with open(next(d for d in Path(__file__).resolve().parents if d.name=='input') / "inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    