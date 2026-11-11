'''
Author:      Kartikay Shukla
File:        inputBDS01a.py
Created:     October 29, 2026
LM:          November 03, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for Beam-Deflection-Slope verification # 1- Cantilever beam with point moment load at the free end XZ plane.
'''

import json
import numpy as np 
from pathlib import Path

NEL = 10
NNPEL = 10
k = 5/6

L = 100 # in
b = 1 # in
h = 1 # in
A3 = b * h
A1 = k * A3
A2 = k * A3
I1 = b * h**3 / 12
I2 = b**3 * h / 12
J = I1 + I2

E = 30E6
nu = 0.25
G = E / 2 / (1+nu)

inputCond = {'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 10,
        'convergence': 1E-5,
        'loadsteps': 1,
        'follower': False,
        }

inputGeom = {'L': L,   # length of beam, in
        'b': b, # width of rod, in
        'h': h, # height of rod, in
        'A1': A1, # area of beam in in^2, of plane E1xE1
        'A2': A2, # area of beam in m^2, of plane E2xE2
        'A3': A3, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': I1, # moment of inertia in^4, bh^3/12, inertia about E1xE1
        'I2': I2, # moment of inertia in^4, b^3h/12, inertia about E2xE2
        'J': J, # polar moment of inertia in^4, inertia about E3xE3, J=I1+I2
        'intCurvature': {0.0: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]},
                        100.0: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]}}
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0], [6]], # [node number], [no of boundary values at node]    
        'Values': [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }
inputNBC = {'globalNode#': [[90], [1]], 
            'Values': [[4, 5]] 
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
    