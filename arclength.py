'''
Author:      Kartikay Shukla
File:        ArcLength.py
Created:     September 23, 2026 
LM:          November 03, 2026

Description
This file contains code for implementation of the arc length method for tracking buckling behavior.
'''
from utilities import *
import warnings
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs, eigsh


class Arclength:
    '''
    Class to solver the for the discplacement of a beam using cylindrical arc length method.
    '''
    def __init__(self, beamSolver):
        '''
        This function initialises variables that will be needed for calculation using arc length method.
        '''
        self.beamSolver = beamSolver
        self.countALIteration = 0 #self.beamSolver.diagnostics['converged']['iteration']
        # Initial allocations from beam solver class
        self.ultimateConvergedSoln = np.array(self.beamSolver.diagnostics['converged']['solution'][-1], dtype=np.float64).ravel()
        self.penUltimateConvergedSoln = np.array(self.beamSolver.diagnostics['converged']['solution'][-2], dtype=np.float64).ravel()    
        self.ultLSIncConvergedSoln = self.determineDelSolnState(self.ultimateConvergedSoln, self.penUltimateConvergedSoln) # incremental solution state from last loadstep to current loadstep, with right update procedure penUltimateConvergedState + Incremental solution state = ultimateConvergedState
        self.ultimateIncConvergedSoln = np.array(self.beamSolver.diagnostics['converged']['incSolution'][-1], dtype=np.float64).ravel()
        self.ultimateLoadRatio = np.array(self.beamSolver.diagnostics['converged']['loadratioHistory'][-1], dtype=np.float64)
        self.penUltimateLoadRatio = np.array(self.beamSolver.diagnostics['converged']['loadratioHistory'][-2], dtype=np.float64)
        # iteration storing arrays
        self.totalSolnState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64) # total solution at current iteration current loadstep
        self.incSolnState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64) # incremental solution at current iteration current loadstep
        self.totalLoadRatio = 0.0
        self.delSolnState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64) # change in solution state, vector from converged solution to next possible state after current iteration current loadstep
        self.delLoadRatio = 0.0
        self.prevDelSolnState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64) # change in solution state, vector from converged solution to next possible state after previous iteration previous loadstep
        self.prevDelLoadRatio = 0.0
        self.iterativeSolnState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64) # change in solution state due to multiple iterations, it changes the delSolnState
        self.constraint = 0.0 # scalar constraint, dot product of external force vector of a converged state with itself, same as extForce2 in predictSolution()
        # storing global stiffness arrays when needed
        self.Kt = np.zeros(shape=[self.beamSolver.shapeGM, self.beamSolver.shapeGM], dtype=np.float64)
        self.residual = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64)
        self.extForce = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64)
        #----------------------------- Preprocessing---------------------------------#
        # scaling array
        self.scale = np.ones(shape=self.beamSolver.shapeGM, dtype=np.float64)
        self.scale[3::self.beamSolver.DOFPN] *= self.beamSolver.h # alternate DOFs (dXi/ds) scaled by h.
        self.scale[4::self.beamSolver.DOFPN] *= self.beamSolver.h
        self.scale[5::self.beamSolver.DOFPN] *= self.beamSolver.h
        self.arcLength = np.linalg.norm(self.scale * self.ultLSIncConvergedSoln) # length of curve in the previous loadstep solution
        #--------------------- Predefined harcoded parameters------------------------#
        self.arcLengthMax = self.arcLength * 2
        self.arcLengthMin = self.arcLength / 50
        self.reduceAL = False
        self.increaseAL = False
        self.restartALSolver = True
        self.stopALSolver = False


    def cylindricalFunction(self, solnInt, solnExt):
        '''
        This function computes solution the coefficient of the cylindrical arc length quadratic equation.
        The quadratic equation is then solved to determine the possible iterative load ratios.
        Arguments:
            solnInt: np.array, iterative incremental solution due to internal forces
            solnExt: np.array, iterative incremental solution due to external force
        '''

        # updateState = self.updateProcedure(solnInt, self.prevDelSolnState)
        updateState = solnInt + self.prevDelSolnState # iterative solution internal + predictive solution

        a1 = (solnExt * self.scale) @ (solnExt * self.scale)
        a2 = 2 * (solnExt * self.scale) @ (updateState * self.scale)
        a3 = (updateState * self.scale) @ (updateState * self.scale) - self.arcLength**2
        a4 = (self.prevDelSolnState * self.scale) @ (solnInt * self.scale) + (self.prevDelSolnState * self.scale) @ (self.prevDelSolnState * self.scale)
        a5 = (self.prevDelSolnState * self.scale) @ (solnExt * self.scale)

        iterativeLR = 0.0
        ignoreLR = 0.0
        discriminant = a2**2 - 4 * a1 * a3
        if discriminant <= 0.0 :
            self.reduceAL = True
        else:
            LR1, LR2 = np.roots([a1, a2, a3])
            cons1 = a4 + a5 * LR1
            cons2 = a4 + a5 * LR2
            if cons2 > cons1:
                iterativeLR = LR2  
                ignoreLR = LR1    
            else:
                iterativeLR = LR1  
                ignoreLR = LR2
        return iterativeLR, ignoreLR

    def thermDynStabilityCheck(self, totalSolnState, prevDelSolnState, totalLoadRatio, iterativeLoadRatio, prevDelLoadRatio, ultimateConvergedSoln, ultimateLoadRatio, iterationInd=1):
        '''
        This function checks the thermodynamic stability of the chosen path (from the loadratio selection), as per the second law of thermodyanmics. 
        The formula for work done is minimised to maximise the entropy change (-TdS).
        Arguments:
            totalSolnState: np.ndarray, total solution state of the beam.
            prevDelSolnState: np.ndarray, previous incremental solution state of the beam, predictor + iterative.
            totalLoadRatio: float, load ratio applied to the beam, total upto the current state
            iterativeLoadRatio: float, iterative load ratio for the current iteration.
            prevDelLoadRatio: float, previous incremental load ratio.
            ultimateConvergedSoln: np.ndarray, ultimate converged solution state.
            ultimateLoadRatio: float, ultimate load ratio.
            iterationInd: int, current iteration number.
        Returns:
            workDone: float, work done for the current iterative load ratio.
        '''

        totalSolnStateMatrix = transformDimension(totalSolnState)
        prevDelSolnStateMatrix = transformDimension(prevDelSolnState)
        Kt, residual, extForce = self.beamSolver.assemble(totalLoadRatio, totalSolnStateMatrix, prevDelSolnStateMatrix, True) # Kt = tangent matrix, residual is total internal and external force from last converged step.
                    
        iterativeSolutionExt = np.asarray(spsolve(Kt, extForce)) # iterative solution due to external load
        iterativeSolutionInt = np.asarray(spsolve(Kt, residual)) # iterative solution due to internal load
        # -----------------------------------update procedure----------------------------------------------------------------------#
        # solution state
        iterativeSolnState = iterativeSolutionInt + iterativeLoadRatio * iterativeSolutionExt
        delSolnState = prevDelSolnState + iterativeSolnState
        totalSS = self.updateProcedure(delSolnState, ultimateConvergedSoln) # ultimateConvergedSoln + delSolnState
        # load ratio
        delLoadRatio = prevDelLoadRatio + iterativeLoadRatio
        totalLR = ultimateLoadRatio + delLoadRatio
        # -----------------------------------convergence----------------------------------------------------------------------#
        totalSSMatrix = transformDimension(totalSS)
        delSolnStateMatrix = transformDimension(delSolnState)
        
        _, residual, _ = self.beamSolver.assemble(totalLR, totalSSMatrix, delSolnStateMatrix, True) # residual is total internal and external force from last converged step.
        workDone = 0.5 * residual @ totalSS
        # print(f"Work done with load ratio {iterativeLoadRatio} is {workDone}.")
        return workDone

    def changeAL(self, totalLoadRatio, step):
        '''
        This function modifies the arc length to accelerate or decelerate solver. It either increases the length of the arc to accelerate solver or decreases arc length to decelerate it. 
        Arguments are for printing. Accelerating the solver may lead to divergence. Code includes segment to increase arc length but only by a small amount (if iterations <=2, then increase arc length otherwise don't.)

        '''

        if self.reduceAL and self.arcLength > self.arcLengthMin:
            print(f'Reducing arc length from {self.arcLength} for Loadratio = {totalLoadRatio}, iteration = {step}')
            self.arcLength *= 0.5  # np.sqrt(self.beamSolver.loopIterations / 2 / self.countALIteration) 
            if self.arcLengthMin >= self.arcLength:
                self.arcLength = self.arcLengthMin
            self.reduceAL = False
            self.restartALSolver = True
        elif self.reduceAL:
            print(f'Reached arc length lower limit. Stopping arc length solver.')
            self.stopALSolver = True
        elif self.increaseAL:
            print(f'Increasing arc length from {self.arcLength} for Loadratio = {totalLoadRatio}, iteration = {step}')
            self.arcLength *= np.sqrt(6 / self.countALIteration) # desired iterations of 4. 
            if self.arcLengthMax <= self.arcLength:
                self.arcLength = self.arcLengthMax
            self.increaseAL = False
             

    def updateProcedure(self, delState:np.ndarray, previousState:np.ndarray)->np.ndarray:
        '''
        This function updates the arc length solver procedure from given arguments. This update procedure is similar to self.beamSolver.configurationUpdate().
        Arguments:
            delState: np.ndarray, of size totalNodes * DOFPN (shapeGM), storing incremental state.
            previousState: np.ndarray, of size totalNodes * DOFPN (shapeGM), storing previous state.
        Returns:
            newState: np.ndarray, of size totalNodes * DOFPN (shapeGM), storing updated solution state.
        '''

        newState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64)

        newState[0::self.beamSolver.DOFPN] = previousState[0::self.beamSolver.DOFPN] + delState[0::self.beamSolver.DOFPN]
        newState[1::self.beamSolver.DOFPN] = previousState[1::self.beamSolver.DOFPN] + delState[1::self.beamSolver.DOFPN]
        newState[2::self.beamSolver.DOFPN] = previousState[2::self.beamSolver.DOFPN] + delState[2::self.beamSolver.DOFPN]

        dummy = np.array([3,4,5], dtype=np.int32) # dummy array used for indexing 

        for node in range(self.beamSolver.totalNodes):
            index = node * self.beamSolver.DOFPN + dummy
            # previous state
            prevRotVec = previousState[index]
            prevRotMat = Rotation.from_rotvec(prevRotVec).as_matrix()
            # incremental state
            delRotVec = delState[index]
            delRotMat = Rotation.from_rotvec(delRotVec).as_matrix()
            # total state
            newRotMat = delRotMat @ prevRotMat
            newRotVec = Rotation.from_matrix(newRotMat).as_rotvec()
            newState[index] = newRotVec

        for i in range(3):
            self.beamSolver.translation[:,i] = newState[i::self.beamSolver.DOFPN]
            self.beamSolver.rotation[:,i] = newState[i+3::self.beamSolver.DOFPN]

        return newState

    def determineDelSolnState(self, newState:np.ndarray, previousState: np.ndarray):
        '''
        This function determines the difference in the two states to compute incremental state. 
        It is similar to updateProcedure(). It undoes what updateProcedure() does.
        Arguments:
            newState: np.ndarray, of size totalNodes * DOFPN (shapeGM), storing updated solution state.
            previousState: np.ndarray, of size totalNodes * DOFPN (shapeGM), storing previous state.
        Returns:
            delState: np.ndarray, of size totalNodes * DOFPN (shapeGM), storing incremental state.
        '''

        delState = np.zeros(shape=self.beamSolver.shapeGM, dtype=np.float64)

        delState[0::self.beamSolver.DOFPN] = newState[0::self.beamSolver.DOFPN] - previousState[0::self.beamSolver.DOFPN] 
        delState[1::self.beamSolver.DOFPN] = newState[1::self.beamSolver.DOFPN] - previousState[1::self.beamSolver.DOFPN]
        delState[2::self.beamSolver.DOFPN] = newState[2::self.beamSolver.DOFPN] - previousState[2::self.beamSolver.DOFPN]

        dummy = np.array([3,4,5], dtype=np.int32) # dummy array used for indexing 

        for node in range(self.beamSolver.totalNodes):
            index = node * self.beamSolver.DOFPN + dummy
            # new state
            newRotVec = newState[index]
            newRotMat = Rotation.from_rotvec(newRotVec).as_matrix()
            # previous state
            prevRotVec = previousState[index]
            prevRotMat = Rotation.from_rotvec(prevRotVec).as_matrix()
            # incremental state
            delRotMat = newRotMat @ prevRotMat.T
            delRotVec = Rotation.from_matrix(delRotMat).as_rotvec()
            delState[index] = delRotVec

        return delState

    def predictSolution(self):
        '''
        This solution calculates the predictor solution vector and load ratio using the last two converged solution vector. It also gives the output of dot product of external force vector (from the converged state) with itself.
        Return:
            predictorSolution: np.ndarray, predicted incremental solution vector.
            predictorLoadRatio: float, predicted load ratio.
        '''
        ultimateConvergedSolnMatrix = transformDimension(self.ultimateConvergedSoln)
        ultimateIncConvergedSolnMatrix = transformDimension(self.ultimateIncConvergedSoln)
        Kt, residual, extForce = self.beamSolver.assemble(self.ultimateLoadRatio, ultimateConvergedSolnMatrix, ultimateIncConvergedSolnMatrix, True) # Kt = sparse tangent matrix, residual is total internal and external force from last converged step.
        
        extForce2 = np.dot(extForce, extForce) # useful for spherical arc length

        predictorSolutionExt = np.asarray(spsolve(Kt, extForce)) # iterative solution due to external load
        predictorSolutionInt = np.asarray(spsolve(Kt, residual)) # iterative solution due to internal load, close to 0 since calculated from converged solution, useful for spherical arc length

        direction = np.sign(np.dot(self.ultLSIncConvergedSoln * self.scale, predictorSolutionExt * self.scale)) # determine direction from previous incremental solution

        predictorLoadRatio = direction * self.arcLength / np.linalg.norm(predictorSolutionExt * self.scale)
        predictorSolution = predictorLoadRatio * predictorSolutionExt

        return predictorSolution, predictorLoadRatio
    
    def iterativeLoop(self, totalSolnState, incSolnState, totalLoadRatio, iterationInd=1):
        '''
        This function contains the steps involved with calculating the iterative solution and load ratio. It is the for loop in the arc length iterative solver. 
        A separate function has been created to restart arc length solver when need be.
        '''
        self.solutionOK = False
        self.totalLoadRatio = totalLoadRatio
        self.totalSolnState[:] = totalSolnState
        self.incSolnState[:] = incSolnState
        for iterations in range(self.beamSolver.loopIterations):
            totalSolnStateMatrix = transformDimension(self.totalSolnState)
            incSolnStateMatrix = transformDimension(self.incSolnState)
            Kt, self.residual[:], self.extForce[:] = self.beamSolver.assemble(self.totalLoadRatio, totalSolnStateMatrix, incSolnStateMatrix) # Kt = tangent matrix, residual is total internal and external force from last converged step.
            
            iterativeSolutionExt = np.asarray(spsolve(Kt, self.extForce[:])) # iterative solution due to external load
            iterativeSolutionInt = np.asarray(spsolve(Kt, self.residual[:])) # iterative solution due to internal load

            iterativeLoadRatio, ignoreLR = self.cylindricalFunction(iterativeSolutionInt, iterativeSolutionExt)
            if self.reduceAL == True:
                break
            # -----------------------------------check path stability----------------------------------------------------------------------#
            '''workDoneIterLR = self.thermDynStabilityCheck(self.totalSolnState, self.totalLoadRatio, iterativeLoadRatio, self.prevDelSolnState, self.prevDelLoadRatio, self.ultimateConvergedSoln, self.ultimateLoadRatio, iterationInd)
            workDoneIgnLR = self.thermDynStabilityCheck(self.totalSolnState, self.totalLoadRatio, ignoreLR, self.prevDelSolnState, self.prevDelLoadRatio, self.ultimateConvergedSoln, self.ultimateLoadRatio, iterationInd)
            if np.abs(workDoneIgnLR) < np.abs(workDoneIterLR):
                print(f"The chosen loadratio maybe for an unstable path. For loadratio {ignoreLR}, work done is {workDoneIgnLR}; for loadratio {iterativeLoadRatio}, work done is {workDoneIterLR}")
            if self.reduceAL == True:
                break'''
            # -----------------------------------update procedure----------------------------------------------------------------------#
            # solution state
            self.iterativeSolnState[:] = iterativeSolutionInt + iterativeLoadRatio * iterativeSolutionExt
            self.delSolnState[:] = self.prevDelSolnState + self.iterativeSolnState # predictor solution + iterative solution 
            self.incSolnState[:] = self.iterativeSolnState # store iterative solution state
            self.totalSolnState[:] = self.updateProcedure(self.delSolnState, self.ultimateConvergedSoln)
            # load ratio
            self.delLoadRatio = self.prevDelLoadRatio + iterativeLoadRatio # predictor loadratio + iterative loadratio
            self.totalLoadRatio = self.ultimateLoadRatio + self.delLoadRatio
            # for next iteration
            self.prevDelSolnState[:] = self.delSolnState
            self.prevDelLoadRatio = self.delLoadRatio
            # -----------------------------------convergence----------------------------------------------------------------------#
            self.check1 = convergenceCheck(self.iterativeSolnState, self.delSolnState)
            totalSolnStateMat = transformDimension(self.totalSolnState)
            incSolnStateMat = transformDimension(self.incSolnState)
            Kt, self.residual[:], self.extForce[:] = self.beamSolver.assemble(self.totalLoadRatio, totalSolnStateMat, incSolnStateMat, True) # residual is total internal and external force from last converged step.
            self.check2 = convergenceCheck(self.residual, self.extForce)
            if (self.check1 < self.beamSolver.convergenceCriteria) and (self.check2 < self.beamSolver.convergenceCriteria) :
                self.solutionOK = True
                self.countALIteration = iterations+1
                self.increaseAL = True if self.countALIteration <= 2 else False 
                self.penUltimateConvergedSoln[:] = self.ultimateConvergedSoln
                self.ultimateIncConvergedSoln[:] = self.iterativeSolnState
                self.ultLSIncConvergedSoln[:] = self.delSolnState
                self.ultimateConvergedSoln[:] = self.totalSolnState
                self.penUltimateLoadRatio = self.ultimateLoadRatio
                self.ultimateLoadRatio = self.totalLoadRatio
                break
            elif iterations + 1 == self.beamSolver.loopIterations:
                self.countALIteration = iterations
                self.reduceAL = True

    def arcLengthSolver(self, step):
        '''
        Steps below were taken from Chapter 9 NLFEM of Solids and structures Part I MA Crisfield
        1. from the last two converged solution, calculate difference to get old/previous incremental solution.
        2. Determine last loadratio.
        3. compute external force vector - its the component of residual vector dependent on loadratio - form is lambda * q where q is the external force vector
            Compute this for the converged for every loadratio computation within the while loop. (tangent term has a component of this.)
            Compute Q (scalar) = q^T.q. This remains fixed for a the following for loop.
        4. compute tangent stiffness matrix from last converged load ratio and last total solution(converged solution = last iterative solution).
        5. compute iterative solution due to external force using - using output of step 3 and 4.
        6. Decide the direction of load ratio from previous incremental solution state (step 1).
        7. Using step 5, and arc length, compute predictor incremental load ratio.
        8. Using step 5 and 7, compute incremental predictor solution.
        9. Now start iteration loop
            9a. compute residual from last displcaement and loadratio (for first iteration this would be predictor solution and laod ratio from step 7 8)
            9b. compute external force vector - its the component of residual vector dependent on loadratio - form is lambda * q where q is the external force vector
                (for first iteration this would be predictor solution and laod ratio from step 7 8)
            9c. compute tangent matrix from last laodratio and previous solution (for first iteration this would be predictor solution and laod ratio from step 7 8)
            9d. Compute iterative solution due to (i) internal and (ii) external force using step 3, 9c and (i) 9a and (ii) 9b.
            9e. Compute a1, a2, a3, a4 and a5. 
            9f. Solve and decide the iterative loadratio. See Figure 9.12. 
            9g. Compute incremental solution and load ratio using 9d, 9f, 7 and 8. This is used in 9e for susequent iterations.
            9h. Update the total state. This is used in 9a 9b and 9c for susequent iterations.
            9i. Check for convergence. If not, continue iteration. If max iteration, reduce arc length and start from the last 2 converged state. If arc length < min arc lengt, STOP.
                If converged, update converged solution and loadratio array. Compute diagnostics. Break loop.
        The above  procedure continue within the while load ratio loop until loadratio >= 1.0 or the maximum number of steps have been reached..
        '''
        self.restartALSolver = True # initialise restartALsolver for every new call of arcLengthSolver().
        recall = (self.beamSolver.rotationMatrixGP.copy(), self.beamSolver.curvatureStrain.copy())
        while self.restartALSolver :
            self.restartALSolver = False
            self.beamSolver.rotationMatrixGP[:], self.beamSolver.curvatureStrain[:] = recall
            predictorSolnState, predictorLoadRatio = self.predictSolution()
            self.prevDelSolnState[:] = predictorSolnState
            self.prevDelLoadRatio = predictorLoadRatio
            self.totalSolnState[:] = self.updateProcedure(predictorSolnState, self.ultimateConvergedSoln)
            self.incSolnState[:] = predictorSolnState
            self.totalLoadRatio = self.ultimateLoadRatio + predictorLoadRatio # ultimate as in last, not final.
            self.iterativeLoop(self.totalSolnState, self.incSolnState, self.totalLoadRatio, step)
            self.changeAL(self.totalLoadRatio, step)
        # self.beamSolver.freshHandoff = False

            
            


            
        


            