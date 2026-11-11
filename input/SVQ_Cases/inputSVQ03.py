'''
Author:      Kartikay Shukla
File:        inputSVQ03.py
Created:     November 04, 2026
LM:          November 04, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for clamped hinged deep circular arch subject to point load at center in XZ plane from Simo VuQuoc's paper II Example7.3.
Assemed circular crossection, SI units and steel material.
'''

import json
import numpy as np
from pathlib import Path

NEL = 40
NNPEL = 2
k = 0.9

E = 210E9 # Pa
nu = 0.3
G = E / 2 / (1+nu)

alpha = 215 # degrees, subtending angle
ratio = 8.97
Pcr = 897 # N


archRadius = 100  # m
L = archRadius * np.deg2rad(alpha) # m
dia = (Pcr * archRadius**2 * 64 / E / ratio / np.pi) ** 0.25 # backcalculating, see DeDeppo Schimdt (1975) for formula.
d = dia # m, 
A3 = np.pi * d**2 / 4
A1 = k * A3
A2 = k * A3
I1 = np.pi * d**4. / 64
I2 = np.pi * d**4. / 64
J = I1 + I2

P = -1000 # Newtons

# compute 

inputCond = {'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 5000,
        'convergence': 1E-6,
        'loadsteps': 155,
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
        'intCurvature': {0.0: {'rotationAngle': alpha * np.pi / 180 / 2, 'rotationVector': [0, 1, 0]},
                    NEL*(L/NEL): {'rotationAngle': -alpha * np.pi / 180 / 2, 'rotationVector': [0, 1, 0]}} # subtending angle of the arc, 215degrees.
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0, 40], [5, 6]], # [node number], [no of boundary values at node]    
        'Values': [[0, 0], [1, 0], [2, 0], [3, 0], [5, 0],  # constraints at DOF 1, 3, and 5 to ensure deformation is in plan
                [0,0],[1,0],[2,0],[3,0],[4,0],[5,0]] # [DOF #, value] # node 0: clamped, node 40 hinged
        }
inputNBC = {'globalNode#': [[20], [1]], # point load at apex/crown of arch
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
    