'''
Author:      Kartikay Shukla
File:        Verification_GD.py
Created:     October 27, 2026
LM:          October 30, 2026

Description
This is a file  to run verification of Simo Vu-Quoc beam model using linear SFBM case.
Please ensure that this file is located in the same directory as solver.py file.

Values for analytical calculation are hardcoded in the results(). 
'''

import subprocess
from utilities import *
from solver import *

def results(Case: str): 
    '''
    This function contains values taken from the dissertation.
    Plotdigitizer tool was used to get these values. The numerical accuracy of the values from the tool maybe compormised.
    '''
    # memory allocation
    deflY = np.zeros(shape=10, dtype=np.float64)
    deflZ = np.zeros(shape=10, dtype=np.float64)

    match Case:
        case '01':
            q = np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0])
            deflY = np.array([1.3072606778968265, 2.5011998948997167, 3.491244562520071, 4.299348962134704, 4.9450851020348585, 5.455452077190319, 5.855393688144104, 6.192555396607596, 6.477540653373427, 6.703015794237001])
            deflZ = np.array([0.10566081803053747, 0.3800893352407088, 0.7647563717046681, 1.1878667561965375, 1.6080343327591744, 2.0035383762006247, 2.3751729775494117, 2.7124748197238198, 3.013668875718915, 3.2873500131375355]) * (-1)  
        case '02':
            q = np.arange(1,15) * 7000 / 14
            deflY = np.array([0.0030031178867231585, 0.0013676619675204193, 0.3391227835268464, 0.39977811358289184, 0.40091623787754166, 0.3859771610015494, 0.36736548135962815, 0.3485529562539452, 0.3321123204345913, 0.31626465693681977, 0.30396526329883894, 0.2921440731460051, 0.28096367566326824, 0.2711605042177548])
            deflZ = np.array([0.002324068937814419, 0.002218864171082079, 0.18542818340060063, 0.3474339600987012, 0.4549436676294497, 0.5253160925036822, 0.5754318177470877, 0.6122534861034068, 0.6399988523116357, 0.6629717477380975, 0.6811626083130894, 0.6961590696072993, 0.7093096654488419, 0.721226496298705]) * (-1)
            print('Obtain solution from Timoshenko-Gere Theory of elastic stability for a closer match. Beam model cannot capture plate model behavior.')
        case '03':
            q = np.array([3.18, 2.94, 2.70, 2.46, 2.22, 1.97, 1.73, 1.50, 1.27, 1.05, 0.84, 0.65, 0.47, 0.32, 0.19, 0.08])
            u3A = np.array([13.766200144044538, 13.358359449516282, 12.87220913709536, 12.38605882467444, 11.892375372277266, 11.365581141844938, 10.819866466821093, 10.188484223230102, 9.511202382574504, 8.721449010180445, 7.827983571136589, 6.7966441515971425, 5.6503805500944075, 4.417748622817434, 3.0374321141455627, 1.5646156541373872])
            u3B = np.array([17.41521811068071, 16.962353765596713, 16.454830358359448, 15.938722675335292, 15.399139625873513, 14.767582192980749, 14.170186673933777, 13.506394409514725, 12.752204465380647, 11.854008915188912, 10.866291631791018, 9.76224865201565, 8.473556148171218, 7.205010414030716, 5.8541257080567615, 4.461370758958986])
            deflY = (u3A + u3B) / 2
            deflZ = np.zeros(shape=len(q))    
            print('Solution may not match. Beam model cannot capture plate model behavior.')    
    return q, deflY, deflZ # type: ignore

filenames = ['inputGD01', 'inputGD02', 
            'inputGD03'
            ]

directory = "input/Gregs_Dissertation"

print('---------------------------------------------------------------------------------------------------------------------------------')
for f, filename in enumerate(filenames):
    # run finite element model    
    subprocess.run(['python', f'{filename}.py'], cwd=directory, check=True)
    modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, followerLoad = inputJson()
    classSVQ = SVQ(modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, followerLoad)
    convergence = classSVQ.solverFEM()
    # run analytical model
    case = filename[-2:]
    q, deflY, deflZ = results(case) # type: ignore
    if case == '01':
        factor = 4
    elif case == '02':
        factor = 7000
    else:
        factor = 3.2

    loadRatioHistory = np.array(classSVQ.diagnostics['converged']['loadratioHistory']) * factor
    solnHistory = classSVQ.diagnostics['converged']['solution']

    tip = np.array([s[-1][:3] for s in solnHistory]) - initialPosition(classSVQ.globalCoords, classSVQ.initialRotations)[-1]
    o = np.argsort(q)
    translationY, translationZ = np.interp(q[o], loadRatioHistory, tip[:,1]), np.interp(q[o], loadRatioHistory, tip[:,2])
    errorDeflY = np.linalg.norm(translationY - deflY[o]) / np.linalg.norm(deflY[o])
    errorDeflZ = np.linalg.norm(translationZ - deflZ[o]) / np.linalg.norm(deflZ[o]) if np.any(deflZ) else 0.0
    
    if errorDeflY <= 1E-2 and errorDeflZ <= 1E-2: # 1% RMSE check
        result = 'Pass'
    else:
        result = 'Fail'

    print(f'Result for case {filename}: {result}.\nRMSE in displacement along X: {errorDeflY} and displacement along Z: {errorDeflZ}')
    print('---------------------------------------------------------------------------------------------------------------------------------')



    
    








