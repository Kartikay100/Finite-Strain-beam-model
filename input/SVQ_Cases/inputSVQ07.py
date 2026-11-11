'''
Author:      Kartikay Shukla
File:        inputSVQ07.py
Created:     November 05, 2026
LM:          November 05, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for lateral buckling of a hinged right angle frame from Simo VuQuoc's paper II Example 7.6.
Refer also Argyris Symeonidis 1981 - Nonlinear finite element analysis of elastic systems under nonconservative loading - natural formulation, Part I.
'''

import json
import numpy as np
from pathlib import Path

NEL = 20
NNPEL = 3
k = 5/6

E = 71240 # N/mm2, aluminum alloy.
nu = 0.31
G = E / 2 / (1+nu)

L = 480 # 
b = 0.6
h = 30
A3 = b * h
A1 = k * A3
A2 = k * A3
I2 = b * h**3 / 12
I1 = b**3 * h / 12
J =  h * b**3 / 3 * (1 - 0.63 * b / h) # See Timoshenko Gere Theory of Elasticity

M = 700 # 

# compute 

inputCond = {'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 11000,
        'convergence': 1E-6,
        'loadsteps': 25,
        'follower': False,
        }

inputGeom = {'L': L,   # length of beam, m
        'b': b, # width of rod, m
        'h': h, # height of rod, m
        'A1': A1, # area of beam in m^2, of plane E1xE1
        'A2': A2, # area of beam in m^2, of plane E2xE2
        'A3': A3, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': I1, # moment of inertia m^4, pi*d^4/64, inertia about E1xE1
        'I2': I2, # moment of inertia m^4, pi*d^4/64, inertia about E2xE2
        'J': J, # polar moment of inertia m^4, inertia about E3xE3, J
        'intCurvature': {0.0: {'rotationAngle': np.pi/4, 'rotationVector': [0, 1, 0]},
                L/2: {'rotationAngle': np.pi/4, 'rotationVector': [0, 1, 0]},
                L/2 + 1E-9: {'rotationAngle': -np.pi/4, 'rotationVector': [0, 1, 0]},
                L: {'rotationAngle': -np.pi/4, 'rotationVector': [0, 1, 0]}} # right angle frame
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0, 20, 40], [4, 1, 4]], # [node number], [no of boundary values at node]    
        'Values': [[0, 0], [1, 0], [3, 0], [5, 0], 
                [2, 0], 
                [0, 0], [1, 0], [3, 0], [5, 0],] # [DOF #, value] # node 0: hinged, node 40 hinged
        }

inputNBC = {'globalNode#': [[0, 20, 40], [1, 1, 1]], # point load 
            'Values': [[4, M], [1, 1E-3], [4, -M]] # small pertubration at apex
        }

inputForce = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx
        }
inputMoment = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }

inputFollowerLoad = {
        'NBC': {'globalNode#': [[0], [0]], 
            'Values': [[0, 0]] 
        },
        'inputForce': {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx
        }
}

with open(next(d for d in Path(__file__).resolve().parents if d.name=='input') / "inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment, inputFollowerLoad], write_file, indent=2)
    