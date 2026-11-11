'''
Author:      Kartikay Shukla
File:        inputSVQ01.py
Created:     November 02, 2026
LM:          November 02, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for cantilever beam with point moment load at the free end XZ plane from Simo VuQuoc's paper II Example7.1.
'''

import json
import numpy as np
from pathlib import Path

NEL = 5
NNPEL = 2
k = 0.9

E = 210E9 # Pa
nu = 0.3
G = E / 2 / (1+nu)

L = 1 # m
d = 3.73E-3 # m
A3 = np.pi * d**2 / 4
A1 = k * A3
A2 = k * A3
I1 = 2 / E
I2 = 2 / E
J = I1 + I2

m = 8 * np.pi # N-m

inputCond = {'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 5000,
        'convergence': 1E-5,
        'loadsteps': 10,
        'follower': False,
        }

inputGeom = {'L': L,   # length of beam, m
        'd': d, # width of rod, m
        'A1': A1, # area of beam in m^2, of plane E1xE1
        'A2': A2, # area of beam in m^2, of plane E2xE2
        'A3': A3, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': I1, # moment of inertia m^4, pi*d^4/64, inertia about E1xE1
        'I2': I2, # moment of inertia m^4, pi*d^4/64, inertia about E2xE2
        'J': J, # polar moment of inertia m^4, inertia about E3xE3, J=I1+I2
        'intCurvature': {0.0: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]},
                    L: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]}}
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0], [6]], # [node number], [no of boundary values at node]    
        'Values': [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }
inputNBC = {'globalNode#': [[5], [1]], 
            'Values': [[4, m]] 
        }

inputForce = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx
        }
inputMoment = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }

inputFollowerLoad = {
        'type': 'NBC', 
        'NBC': {'globalNode#': [[0], [0]], 
            'Values': [[0, 0]] 
        },
        'inputForce': {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx
        }
}

with open(next(d for d in Path(__file__).resolve().parents if d.name=='input') / "inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment, inputFollowerLoad], write_file, indent=2)
    