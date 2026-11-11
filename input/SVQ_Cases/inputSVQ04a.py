'''
Author:      Kartikay Shukla
File:        inputSVQ04a.py
Created:     November 04, 2026
LM:          November 04, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for buckling of a hinged right angleframe under fixed load in XZ plane from Simo VuQuoc's paper II Example7.4.
'''

import json
import numpy as np
from pathlib import Path

NEL = 10
NNPEL = 3
k = 5/6

E = 7.2E6 # maybe aluminum or composite material.
nu = 0.3
G = E / 2 / (1+nu)

L = 240 # 
A3 = 6
I2 = 2
h = (12 * I2 / A3) ** 0.5
b = A3 / h


A1 = k * A3
A2 = k * A3
I1 = b * h**3 / 12
J = I1 + I2

P = -40_000 # Newtons

# compute 

inputCond = {'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 5000,
        'convergence': 1E-6,
        'loadsteps': 43,
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
        'J': J, # polar moment of inertia m^4, inertia about E3xE3, J=I1+I2
        'intCurvature': {0.0: {'rotationAngle': np.pi/2, 'rotationVector': [0, 1, 0]},
                L/2: {'rotationAngle': np.pi/2, 'rotationVector': [0, 1, 0]},
                L/2 + 1E-9: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]},
                L: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]}} # subtending angle of the arc, 215degrees.
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0, 20], [5, 5]], # [node number], [no of boundary values at node]    
        'Values': [[0, 0], [1, 0], [2, 0], [3, 0], [5, 0],  # constraints at DOF 1, 3, and 5 to ensure deformation is in plane.
                [0,0],[1,0],[2,0],[3,0], [5,0]] # [DOF #, value] # node 0: hinged, node 40 hinged
        }
inputNBC = {'globalNode#': [[12], [1]], # point load at apex/crown of arch
            'Values': [[0, P]] 
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
    