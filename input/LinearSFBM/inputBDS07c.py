'''
Author:      Kartikay Shukla
File:        inputBDS07c.py
Created:     October 30, 2026
LM:          November 03, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for Beam-Deflection-Slope verification # 7c- Pinned-roller supported beam with UVL, maximum at free end, off-plane.
'''

import json
import numpy as np 
from pathlib import Path

NEL = 10
NNPEL = 10
k = 0.9

L = 100 # in
d = 1 # in
A3 = np.pi * d**2 / 4
A1 = k * A3
A2 = k * A3
I1 = np.pi * d**4 / 64
I2 = np.pi * d**4 / 64
J = I1 + I2

E = 30E6
nu = 0.25
G = E / 2 / (1+nu)

# UVL
totalNodes = NEL * (NNPEL - 1) + 1
load = 1 / 100 / np.sqrt(2) # lb/in
z = [0.0, 0.40233045916771015, 1.3061306744724854, 2.610375250947769, 4.173605211668065, 5.826394788331935, 7.389624749052226, 8.693869325527519, 9.597669540832296, 10.0, 10.40233045916771, 11.306130674472486, 12.610375250947769, 14.173605211668065, 15.826394788331935, 17.389624749052228, 18.69386932552752, 19.597669540832296, 20.0, 20.402330459167707, 21.306130674472488, 22.61037525094777, 24.173605211668065, 25.826394788331935, 27.389624749052228, 28.69386932552752, 29.597669540832293, 30.0, 30.40233045916771, 31.306130674472485, 32.61037525094777, 34.17360521166806, 35.82639478833194, 37.38962474905223, 38.693869325527515, 39.5976695408323, 40.0, 40.40233045916771, 41.30613067447249, 42.61037525094777, 44.17360521166806, 45.82639478833194, 47.389624749052224, 48.693869325527515, 49.59766954083229, 50.0, 50.40233045916771, 51.306130674472485, 52.61037525094776, 54.17360521166806, 55.82639478833194, 57.38962474905223, 58.693869325527515, 59.59766954083229, 60.0, 60.402330459167715, 61.306130674472485, 62.61037525094777, 64.17360521166806, 65.82639478833194, 67.38962474905223, 68.69386932552752, 69.5976695408323, 70.0, 70.4023304591677, 71.30613067447248, 72.61037525094777, 74.17360521166808, 75.82639478833194, 77.38962474905223, 78.69386932552752, 79.5976695408323, 80.0, 80.4023304591677, 81.30613067447248, 82.61037525094777, 84.17360521166806, 85.82639478833194, 87.38962474905223, 88.69386932552752, 89.59766954083229, 90.0, 90.40233045916771, 91.30613067447248, 92.61037525094775, 94.17360521166808, 95.82639478833192, 97.38962474905223, 98.6938693255275, 99.5976695408323, 100.0]
UVL = [[]] * totalNodes * 2
for i in range(totalNodes):
    UVL[2*i] = [0, load*z[i]]
    UVL[2*i + 1] = [1, load*z[i]]

inputCond = {'NEL': NEL,
        'NNPEL': NNPEL,
        'iterations': 20,
        'convergence': 1E-5,
        'loadsteps': 1,
        'follower': False
        }

inputGeom = {'L': L,   # length of beam, in
        'd': d, # diameter of rod, in
        'A1': A1, # area of beam in in^2, of plane E1xE1
        'A2': A2, # area of beam in m^2, of plane E2xE2
        'A3': A3, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': I1, # moment of inertia in^4, piD^4/64, inertia about E1xE1
        'I2': I2, # moment of inertia in^4, piD^4/64, inertia about E2xE2
        'J': J, # polar moment of inertia in^4, inertia about E3xE3, J=I1+I2
        'intCurvature': {0.0: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]},
                    100.0: {'rotationAngle': 0.0, 'rotationVector': [0, 1, 0]}}
        }

inputMatProp = {'E': E, # Youngs modulus, psi
                'G': G, # shear modulus, psi
                }

inputEBC = {'globalNode#': [[0, 90], [4, 2]], # [node number], [no of boundary values at node]    
        'Values': [[0,0],[1,0],[2,0],[5,0],[0,0],[1,0]] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }

inputNBC = {'globalNode#': [[0], [0]], 
            'Values': [[0, 0]] 
        }

inputForce = {'globalNode#': [np.array(range(91)).tolist(), np.tile(np.array([2]), 91).tolist()], # [node number], [no of boundary values at node]
            'Values': UVL # [DOF #, value], 0=Tx
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
    