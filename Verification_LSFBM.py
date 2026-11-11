'''
Author:      Kartikay Shukla
File:        Verification_LSFBM.py
Created:     October 27, 2026
LM:          October 30, 2026

Description
This is a file  to run verification of Simo Vu-Quoc beam model using linear SFBM case.
Please ensure that this file is located in the same directory as solver.py file.
BDS: Beam-Deflection-Slope.

Values for analytical calculation are hardcoded in the exactBeamDefl(). 
Only the angle of slope is verified, not its direction.
'''

import subprocess
from utilities import *
from solver import *

def exactBeamDefl(Case: list, EI: np.float64, L: np.float64, Z): 
    '''
    This functions contains exact solution for beam deflection examples. 
    Axes considered in these solutions are Z along the length of the beam, X is the in plane axis along which force is applied and Y is the rotation axis.
    See pictures stored in the verification folder for reference to the configuration and formula.

    Case: It is a list with values required for calculation of each example.
            Example 1: [1, Mo] where 1 is the example number and Mo is the applied moment at the free end of the cantilever beam.
                    This is a cantilever beam with point moment load Mo at the free end.
            Example 2: [2, Mo, a] where 2 is the example number and Mo is the applied moment load. 'a' is the distance from the fixed end of the cantilever beam.
                    This is a cantilever beam with point moment load applied at 'a' from the fixed end.
            Example 3: [3, Po] where 3 is the example number and Po is the applied point force load at the free end of the cantilever beam.
                    This is a cantilever beam with point force load Po at the free end.
            Example 4: [4, Po, a] where 4 is the example number and Po is the applied point force load. 'a' is the distance from the fixed end of the cantilever beam.
                    This is a cantilever beam with point force load applied at 'a' from the fixed end.
            Example 5: [5, po] where 5 is the example number and po is the uniformly distributed force load.
                    This is a simply supported beam with pin and roller supports with distrbuted force load applied on its entire length.
            Example 6: [6, po, a] where 6 is the example number, po is the uniformly distributed force load and 'a' is the length of beam under distrbuted load from the pinned end.
                    This is a simply supported beam with pin and roller supports with distrbuted force load applied on a section of length (a) from the pinned end.
            Example 7: [7, po] where 7 is the example number and po is the maximum of uniformly varying force load, applied at the roller end.
                    This is a simply supported beam with pin and roller supports with uniformly varying force load applied on its entire length, maximum at roller end.
            Example 8: [8, po] where 8 is the example number and po is the maximum of uniformly varying force load, applied at the center of the beam.
                    This is a simply supported beam with pin and roller supports with uniformly varying force load applied on its entire length, maximum at the center of beam.
            Example 9: simply supported beam with pinned-roller supported ends with parabolically varying load. Result not included in this function.
            Example 10: [10, po] where 10 is the example number and po is the uniformly distributed force load.
                    This is a cantilever beam with distributed force load.
            Example 11: [11, po, a] where 11 is the example number, po is the uniformly distributed force load and 'a' is the length of beam under distributed load from fixed end.
                    This is a cantilever beam with distributed force load over a section of length 'a' from fixed end.
            Example 12: [12, po] where 12 is the example number, po is the uniformly varying force load, maximum at the free end.
                    This is a cantilever beam with uniformly varying distributed force load over the length of the beam, maximum at the free end.
            Example 13: [13, po] where 13 is the example number, po is the uniformly varying force load, maximum at the fixed end.
                    This is a cantilever beam with uniformly varying distributed force load over the length of the beam, maximum at the fixed end.
            Example 14: cantilever beam with parabolically varying load. Result not included in this function.
    EI: material property. E = Young's modulus, I = moment of Inertia.
    L: length of beam
    Z: coordinates along the beam. lenth gives the total number of nodes.
    '''
    
    arraySize = len(Z)
    # memory allocation
    deflection = np.zeros(shape=arraySize, dtype=np.float64)
    slope = np.zeros(shape=arraySize, dtype=np.float64)
    moment = np.zeros(shape=arraySize, dtype=np.float64)
    force = np.zeros(shape=arraySize, dtype=np.float64)
    
    number = Case[0]
    match number:
        case '01':
            Mo = Case[1] if len(Case) == 2 else 5 # lb-in
            constant = Mo / EI
            deflection[:] = constant * Z**2 / 2
            slope[:] = constant * Z
            moment[:] = Mo
            force[:] = 0.0
        case '02':
            if len(Case) == 3:
                Mo, a = Case[1], Case[2] 
            else: 
                Mo, a = 5, 60 # lb-in, in
            constant = Mo / EI
            deflection[:] = (constant * Z**2 / 2) * np.heaviside(a-Z,1) + (constant * a / 2)*(2*Z - a) * np.heaviside(Z-a,0)
            slope[:] = (constant * Z) * np.heaviside(a-Z,1) + constant * a * np.heaviside(Z-a,0)
            moment[:] = Mo * np.heaviside(a-Z,1)
            force[:] = 0
        case '03':
            Po = Case[1] if len(Case) == 2 else 50 # lb
            constant = Po / EI
            deflection[:] = constant * Z**2 * (3*L-Z) / 6
            slope[:] = constant * Z * (2 * L - Z) / 2
            moment[:] =  Po * (L - Z)
            force[:] = -Po
        case '04':
            if len(Case) == 3:
                Po, a = Case[1], Case[2] 
            else: 
                Po, a = 50, 60 # lb-in, in
            constant = Po / EI
            deflection[:] = (constant * Z**2 * (3*a-Z) / 6) * np.heaviside(a-Z,1) + (constant * a**2 * (3*Z-a) / 6) * np.heaviside(Z-a,0)
            slope[:] = (constant * Z * (2*a-Z) / 2) * np.heaviside(a-Z,1) + (constant * a**2 / 2) * np.heaviside(Z-a,0)
            moment[:] = -(Po * Z - Po * a) * np.heaviside(a-Z, 1)
            force[:] = -(Po) * np.heaviside(a-Z,1)
        case '05':
            po = Case[1] if len(Case) == 2 else 1 # lb/in
            constant = po / EI / 24
            deflection[:] = constant * Z * ( L**3 - 2 * L * Z**2 + Z**3)
            slope[:] = constant * ( L**3 - 6 * L * Z**2 + 4 * Z**3)
            moment[:] = po * (-L * Z + Z**2) / 2
            force[:] = po * (-L + 2*Z) / 2
        case '06':
            if len(Case) == 3:
                po, a = Case[1], Case[2] 
            else: 
                po, a = 1, 60 # lb-in, in
            constant = po / EI / 24 / L
            deflection[:] = (constant * Z * (a**4 - 4*a**3*L + 4*a**2*L**2 + 2*a**2*Z**2 - 4*a*L*Z**2 + L*Z**3)) * np.heaviside(a-Z,1) \
                            + (constant * a**2 * (-a**2*L + 4*L**2*Z + a**2*Z -6*L*Z**2 + 2*Z**3)) * np.heaviside(Z-a,0)
            slope[:] = (constant * (a**4 - 4*a**3*L + 4*a**2*L**2 + 6*a**2*Z**2 - 12*a*L*Z**2 + 4*L*Z**3)) * np.heaviside(a-Z,1) \
                            + (constant * a**2 * (4*L**2 + a**2 -12*L*Z + 6*Z**2)) * np.heaviside(Z-a,0)
            moment[:] = ((po * (a**2*Z - 2*a*L*Z + L*Z**2)) * np.heaviside(a-Z,1) \
                            + (po * a**2 * (-L + Z)) * np.heaviside(Z-a,0)) / 2 / L
            force[:] = ((po * (a**2 - 2 * a * L + 2 * L * Z)) * np.heaviside(a-Z,1)  \
                            + (po * a**2) * np.heaviside(Z-a,0)) / 2 / L
        case '07':
            po = Case[1] if len(Case) == 2 else 1 # lb/in
            constant = po / 360 / L / EI
            deflection[:] = constant * Z * (7 * L**4 - 10 * L**2 * Z**2 + 3 * Z**4)
            slope[:] = constant * (7 * L**4 - 30 * L**2 * Z**2 + 15 * Z**4)
            moment[:] = po * (- L**2 * Z + Z**3) / 6 / L
            force[:] = po * (- L**2 + 3 * Z**2) / 6 / L
        case '08':
            po = Case[1] if len(Case) == 2 else 1 # lb/in
            constant = po / 960 / L / EI
            deflection[:] = constant * Z * (5 * L**2 - 4 * Z**2) **2 * np.heaviside(L/2-Z,1) \
                            + constant * (L-Z) * (5 * L**2 - 4 * (L-Z)**2) **2 * np.heaviside(Z-L/2,0)
            slope[:] = 5 * constant * (5 * L**2 - 4 * Z**2) * (L**2 - 4 * Z**2) * np.heaviside(L/2-Z,1) \
                            - 5 * constant * (5 * L**2 - 4 * (L-Z)**2) * (L**2 - 4 * (L-Z)**2) * np.heaviside(Z-L/2,0)
            moment[:] = po * ((-3 * L**2 * Z + 4 * Z**3)* np.heaviside(L/2-Z,1) \
                            + (-3 * L**2 * (L-Z) + 4 * (L-Z)**3)* np.heaviside(Z-L/2,0)) / 12 / L
            force[:] = po * ((-L**2 + 4 * Z**2)* np.heaviside(L/2-Z,1) \
                            - (-L**2 + 4 * (L-Z)**2)* np.heaviside(Z-L/2,0)) / 4 / L
        case '10':
            po = Case[1] if len(Case) == 2 else 1 # lb/in
            constant = po / 24 / EI
            deflection[:] = constant * Z**2 * (6 * L**2 -4 * L * Z + Z**2)
            slope[:] = 4 * constant * Z * (3 * L**2 - 3 * L * Z + Z**2)
            moment[:] = po * (L - Z)**2 / 2
            force[:] = po * (-L + Z)
        case '11':
            if len(Case) == 3:
                po, a = Case[1], Case[2] 
            else: 
                po, a = 1, 60 # lb-in, in
            constant = po / 24 / EI
            deflection[:] = (constant * Z**2 * (6 * a**2 - 4 * a * Z + Z**2)) * np.heaviside(a-Z,1) \
                            + (constant * a**3 * (4 * Z - a)) * np.heaviside(Z-a,0)
            slope[:] = 4 * ((constant * Z * (3 * a**2 - 3 * a * Z + Z**2)) * np.heaviside(a-Z,1) \
                            + (constant * a**3) * np.heaviside(Z-a,0))
            moment[:] = po * (a-Z)**2 * np.heaviside(a-Z,1) / 2
            force[:] = po * (-a + Z) * np.heaviside(a-Z,1)
        case '12':
            po = Case[1] if len(Case) == 2 else 1 # lb/in
            constant = po / 120 / L / EI
            deflection[:] = constant * Z**2 * (20 * L**3 - 10 * L**2 * Z + Z**3)
            slope[:] = 5 * constant * Z * (8 * L**3 - 6 * L**2 * Z + Z**3)
            moment[:] = po * (2 * L**3 - 3 * L**2 * Z + Z**3) / 6 / L
            force[:] = po * (-L**2 + Z**2) / 2 / L
        case '13':
            po = Case[1] if len(Case) == 2 else 1 # lb/in
            constant = po / 120 / L / EI
            deflection[:] = constant * Z**2 * (10 * L**3 - 10 * L**2 * Z + 5 * L * Z**2 - Z**3)
            slope[:] = 5 * constant * Z * (4 * L**3 - 6 * L**2 * Z + 4 * L * Z**2 - Z**3)
            moment[:] = po * (L**3 - 3 * L**2 * Z + 3 * L * Z**2 - Z**3) / 6 / L
            force[:] = -po * (L-Z)**2 / 2 / L

    return deflection, slope, moment, force

filenames = ['inputBDS01a', 'inputBDS01b', 'inputBDS01c',
            'inputBDS02a', 'inputBDS02b', 'inputBDS02c',
            'inputBDS03a', 'inputBDS03b', 'inputBDS03c',
            'inputBDS04a', 'inputBDS04b', 'inputBDS04c',
            'inputBDS05a', 'inputBDS05b', 'inputBDS05c',
            'inputBDS06a', 'inputBDS06b', 'inputBDS06c',
            'inputBDS07a', 'inputBDS07b', 'inputBDS07c',
            'inputBDS08a', 'inputBDS08b', 'inputBDS08c',
            'inputBDS10a', 'inputBDS10b', 'inputBDS10c',
            'inputBDS11a', 'inputBDS11b', 'inputBDS11c',
            'inputBDS12a', 'inputBDS12b', 'inputBDS12c',
            'inputBDS13a', 'inputBDS13b', 'inputBDS13c',
            ]

directory = 'input/LinearSFBM'

print('---------------------------------------------------------------------------------------------------------------------------------')
for f, filename in enumerate(filenames):
    # run finite element model    
    subprocess.run(['python', f'{filename}.py'], cwd=directory, check=True)
    modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, followerLoad = inputJson()
    classSVQ = SVQ(modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, followerLoad)
    convergence = classSVQ.solverFEM()
    # run analytical model
    plane, case = filename[-1], filename[-3: -1]
    Z = np.unique(classSVQ.globalCoords.flatten())
    displacement = np.array([np.linalg.norm(classSVQ.translation[node,[0,1]]) for node in range(classSVQ.totalNodes)])
    if plane == 'a':
        vector = np.array([0.0, 1.0, 0.0])
        inertia = classSVQ.I2
        EI = classSVQ.young * inertia
        deflection, slope, moment, force = exactBeamDefl([case], EI, classSVQ.L, Z)
    elif plane == 'b':
        vector = np.array([-1.0, 0.0, 0.0])
        inertia = classSVQ.I1
        EI = classSVQ.young * inertia
        deflection, slope, moment, force = exactBeamDefl([case], EI, classSVQ.L, Z)
    else:
        vector = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])
        inertia = classSVQ.I2 # off-plane case, I1 = I2 for symmetrical crossection.
        EI = classSVQ.young * inertia
        deflection, slope, moment, force = exactBeamDefl([case], EI, classSVQ.L, Z)
    displacement = np.array([classSVQ.translation[node,:] @ np.cross(vector, classSVQ.E3) for node in range(classSVQ.totalNodes)])
    rotationAngle = np.array([classSVQ.rotation[node,:] @ vector for node in range(classSVQ.totalNodes)])
    errorDeflection = np.linalg.norm(displacement - deflection) / np.linalg.norm(deflection)
    errorSlope = np.linalg.norm(rotationAngle - slope) / np.linalg.norm(slope)
    if errorDeflection <= 2E-2 and errorSlope <= 2E-2: # 2% RMSE check
        result = 'Pass'
    else:
        result = 'Fail'
    
    print(f'Result for case {filename}: {result}.\nRMSE in displacement: {errorDeflection} and slope: {errorSlope}')
    print('---------------------------------------------------------------------------------------------------------------------------------')








