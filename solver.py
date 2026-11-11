'''
Author:      Kartikay Shukla
File:        solver.py
Created:     October 27, 2026
LM:          October 27, 2026

Description
This file contains finite element modl for solving beam problems
'''

from utilities import *
from arclength import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

class SVQ():
    '''
    Simo Vu-Quoc beam model.
    '''

    def __init__(self, modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, follower):
        '''
        Class constructor.
        '''
        # inputs
        self.modelingConditions, self.geometry, self.materialProperty, self.essentialBC, self.naturalBC, self.distributedForce, self.distributedMoment, self.follower = modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, follower
        # modeling conditions
        self.NEL = modelingConditions['NEL']
        self.NNPEL = modelingConditions['NNPEL']
        self.maxIterations = modelingConditions['iterations']
        self.convergenceCriteria = modelingConditions['convergence']
        self.loadsteps = modelingConditions['loadsteps']
        self.DOFPN = 6
        self.follower = modelingConditions['follower']
        # geometry of the problem
        self.L = geometry['L']
        # self.d = geometry['d']
        # self.D = geometry['D']
        self.A1 = geometry['A1']
        self.A2 = geometry['A2']
        self.A3 = geometry['A3']
        self.I1 = geometry['I1']
        self.I2 = geometry['I2']
        self.J = geometry['J']
        self.inputRotations = {float(k): v for k, v in geometry['intCurvature'].items()}
        # material properties
        self.young = materialProperty['E']
        self.shear = materialProperty['G']
        #--------------------- Predefined harcoded parameters------------------------#
        # prameters related to seeding the branch
        self.seedParam = {
                        'tanSymmTol': 1E-10, # relative tolerance for checking Kt + Kt^T
                        'numericalConditionTol': 1E9,} # condition of numerical tolerance, max accepted ratio of sigma_max / sigma_min, used for switching from load controlled to arc length solver, kept conservative to allow switching before load control diverges.
        self.initialStep = False
        self.E3 = np.array([0,0,1], dtype=np.float64)
        self.loopIterations = 50
        self.solverType = 'Load-controlled'
        self.arclengthClass = None
        #------------------------- Preprocessing------------------------#
        # for memory allocation
        eqnsPerElem = self.NNPEL * self.DOFPN
        self.elemComponents = eqnsPerElem ** 2 # terms in an element matrix, square matrix of shape eqnsPerElem x eqnsPerElem
        self.totalNodes = self.NEL * (self.NNPEL - 1) + 1
        self.shapeGM = self.totalNodes * self.DOFPN
        sizeSparGM = eqnsPerElem * self.NEL
        self.NGQP = self.NNPEL - 1  if self.NNPEL <= 3 else self.NNPEL
        # model properties
        self.connElem, self.globalCoords, self.h, self.connDOF = mesh1D(self.L, self.NEL, self.NNPEL)
        self.nodalCoords = np.unique(self.globalCoords)
        self.EBC = boundaryHandler(essentialBC, self.connElem, 'EBC') # dictionary of essential boundary condition
        self.NBC = boundaryHandler(naturalBC, self.connElem, 'NBC') # dictionary of natural boundary condition
        self.fixedDOFs = np.array([int(self.connDOF[elem, dof]) for elem, item in self.EBC.items() for dof in item['localDOF_elem#']]) # this creates an array of fixed global DOFs
        self.freeDOFs = np.array([dof for dof in range(self.shapeGM) if dof not in self.fixedDOFs ]) # array of free global DOFs
        # material properties
        listProperties = [self.young, self.shear, self.A1, self.A2, self.A3, self.I1, self.I2, self.J, self.h]
        propertiesArrays = propertyPerElement(self.NEL, listProperties)
        self.E, self.G = propertiesArrays[0], propertiesArrays[1]
        areas = [propertiesArrays[2], propertiesArrays[3], propertiesArrays[4]]
        secondmomentArea = [propertiesArrays[5], propertiesArrays[6], propertiesArrays[7]]
        self.h_elem = propertiesArrays[8]
        self.inConstModForce, self.inConstModMoment = constitutiveMatrix(self.NEL, self.E, self.G, areas, secondmomentArea) # initial constitutive model for force and moment respectively
        # distributed forces
        self.distributedForce = gendistributedForce(distributedForce, self.connElem)
        self.distributedMoment = gendistributedForce(distributedMoment, self.connElem)
        # numerical integration
        self.interpolant, self.interpolantD = lagrangePoly(self.NNPEL, self.NGQP)
        _, self.gaussWt = glQuad(self.NGQP)
        self.jacobian = self.globalCoords @ self.interpolantD
        # parameters for solution
        self.sparRow = np.repeat(self.connDOF, eqnsPerElem).ravel() # indices of rows of sparse matrix for global matrix
        self.sparCol = np.tile(self.connDOF, eqnsPerElem).ravel() # indices of columns of sparse matrix for global matrix
        self.perm = permutationOp()
        # compute initial curvature of the body
        self.initialRotations = initialCurvature(self.globalCoords, self.inputRotations) # compute rotation vector for each global node.
        # compute follower load properties
        if self.follower:
            self.followerNBC = boundaryHandler(follower['NBC'], self.connElem, 'NBC')
            self.followerDistForce = gendistributedForce(follower['inputForce'], self.connElem)
        #------------------------- Memory allocation------------------------#
        # tangent and residuals
        self.intTangentElemPDOF = {f"T{a}{b}": np.zeros(shape=[self.NNPEL, self.NNPEL], dtype=np.float64) for a in range (self.DOFPN) for b in range(self.DOFPN)}
        self.extTangentElemPDOF = {f"T{a}{b}": np.zeros(shape=[self.NNPEL, self.NNPEL], dtype=np.float64) for a in range (self.DOFPN) for b in range(self.DOFPN)}
        self.intResidualElemPDOF = {f"R{a}": np.zeros(shape=self.NNPEL, dtype=np.float64) for a in range (self.DOFPN)}
        self.extResidualElemPDOF = {f"R{a}": np.zeros(shape=self.NNPEL, dtype=np.float64) for a in range (self.DOFPN)}
        self.tangentElem = np.zeros(shape=[eqnsPerElem, eqnsPerElem], dtype=np.float64)
        self.intTangentElem = np.zeros(shape=[eqnsPerElem, eqnsPerElem], dtype=np.float64)
        self.extTangentElem = np.zeros(shape=[eqnsPerElem, eqnsPerElem], dtype=np.float64)
        self.intResidualElem = np.zeros(shape=eqnsPerElem, dtype=np.float64)
        self.extResidualElem = np.zeros(shape=eqnsPerElem, dtype=np.float64)
        self.residualElem = np.zeros(shape=eqnsPerElem, dtype=np.float64)
        self.followerVector = np.zeros(shape=eqnsPerElem, dtype=np.float64)
        # solutions
        self.delConfig = np.zeros(shape=[self.totalNodes, self.DOFPN], dtype=np.float64)
        self.prevConfig = np.zeros(shape=[self.totalNodes, self.DOFPN], dtype=np.float64)
        self.initialCoordinates = initialPosition(self.globalCoords, self.initialRotations, self.inputRotations) # compute nodal coordinates
        self.prevConfig[:,[0,1,2]] = self.initialCoordinates
        self.prevConfig[:,[3,4,5]] = self.initialRotations
        self.newConfig = np.zeros(shape=[self.totalNodes, self.DOFPN], dtype=np.float64)
        self.translation = np.zeros(shape=[self.totalNodes, 3], dtype=np.float64) # current state, from self.prevSoln
        self.rotation = np.zeros(shape=[self.totalNodes, 3], dtype=np.float64) # current state, from self.prevSoln
        self.incRotation = np.zeros(shape=[self.totalNodes, 3], dtype=np.float64) # incremental rotation, from solveMatrixEquation()
        # configuration update
        self.curvatureStrain = np.zeros(shape=[self.NEL, self.NGQP, self.DOFPN//2], dtype=np.float64)
        self.prevCurvatureStrain = np.zeros(shape=[self.NEL, self.NGQP, self.DOFPN//2], dtype=np.float64)
        self.rotationMatrixGP = np.zeros(shape=[self.NEL, self.NGQP, 3, 3], dtype=np.float64)
        self.spuriousStrain = np.zeros(shape=[self.NEL, self.NGQP, 3], dtype=np.float64)
        for nel in range(self.NEL): # compute rotations and strain at Gauss points (Gauss Legendre Points)
            coordGP = self.globalCoords[nel, :] @ self.interpolant # gauss points at element number 'nel'
            vectorGP = initialCurvature(coordGP, self.inputRotations) # calculate rotation vector at gauss points from input rotations, this allows for kink at nodal location
            for GP in range(self.NGQP):
                S1 = self.interpolantD[:,GP] / self.jacobian[nel,GP]
                self.rotationMatrixGP[nel, GP, :, :] = Rotation.from_rotvec(vectorGP[GP]).as_matrix() # compute rotation amtrix at Gauss point
                displacementGP = self.prevConfig[np.ix_(self.connElem[nel],[0,1,2])].T @ S1
                self.spuriousStrain[nel,GP,:] = self.rotationMatrixGP[nel, GP, :, :].T @ displacementGP -  self.E3 # this is in reference configuration
        # diagnostics
        self.diagnostics = {
            'converged': {
            'loadratioHistory':[],
            'iteration': [], # stores total number of iterations required for convergence for each loadstep.
            'solution':[],
            'incSolution':[],
            'residualNormHistory': [], # norm of residual vector for each accepted loadstep, should be close to 0 for an accepted residual - as residual should be minimised with a converged stable solution
            'singHistory': [[],[]], # [0,:] minimum, [1,:] maximum # check for stability based on the condition number of tangent matrix 
            'condNumHistory': [], # 
            'tanSymmHistory': [], # check the symmetry of the tangent matrix, should be close to 0 for a symmetric matrix.
            'stabilityEigenvalueHistory': [],
            'morseHistory': [], # morse index for each accepted loadstep, negative eigen values indicates a loss of stability.
            'minEigValueHistory': [], # size 2: stores minmum and second minmum eigen values of the tangent matrix.
            'workdone':[],
            'error': [],

        }, 
            'diverged': {
            'loadratioHistory':[],
            'iteration': [], # stores current iteration number.
            'solution':[],
            'incSolution':[], # incremental solution after the last iteration
            'residualNormHistory': [], # norm of residual vector for each accepted loadstep, should be close to 0 for an accepted residual - as residual should be minimised with a converged stable solution
            'singHistory': [[],[]], # [0,:] minimum, [1,:] maximum # check for stability based on the condition number of tangent matrix 
            'condNumHistory': [], # 
            'tanSymmHistory': [], # check the symmetry of the tangent matrix, should be close to 0 for a symmetric matrix.
            'stabilityEigenvalueHistory': [],
            'morseHistory': [], # morse index for each accepted loadstep, negative eigen values indicates a loss of stability.
            'minEigValueHistory': [], # size 2: stores minmum and second minmum eigen values of the tangent matrix.
            'workdone':[],
            'error': [],
            } 
        }

    def element(self, i:int, previousSolution:np.ndarray, incrementalSolution:np.ndarray, checkRun=False):
        '''
        This functions computes and assembles the element level stiffness equation. It returns element tangent matrix and residual vector.
        Argument:
            i: int, element #.
            previousSolution: np.ndarray of shape total nodes x DOFPN. It is passed to the element().
            incrementalSolution: np.ndarray of shape total nodes x DOFPN. It is passed to the element().
            checkRun: boolean. Identify if the element level call is for checks or computing new state. Updating previous curvature strain array is dependent on this.
                False: not a check run. Will update curvature strain.
                True: check run. Will not update curvature strain.
        Returns:
            None. Writes in the predefined memory.
        '''
        # re-zero
        self.intTangentElem[:,:] = 0.0
        self.extTangentElem[:,:] = 0.0
        self.intResidualElem[:] = 0.0
        self.extResidualElem[:] = 0.0
        self.residualElem[:] = 0.0
        for key in self.intTangentElemPDOF.keys():
            self.intTangentElemPDOF[key][:,:] = 0.0
            self.extTangentElemPDOF[key][:,:] = 0.0
        for key in self.intResidualElemPDOF.keys():
            self.intResidualElemPDOF[key][:] = 0.0
            self.extResidualElemPDOF[key][:] = 0.0
        self.followerVector[:] = 0.0

        phio = previousSolution[np.ix_(self.connElem[i],[0,1,2])]
        nuo = previousSolution[np.ix_(self.connElem[i],[3,4,5])] # shape NNPEL x 3
        theta = incrementalSolution[np.ix_(self.connElem[i],[3,4,5])]

        self.prevCurvatureStrain[i,:,:] = self.curvatureStrain[i,:,:]
        for GP in range(self.NGQP):
            constant = self.jacobian[i,GP] * self.gaussWt[GP]
            S0 = self.interpolant[:,GP]
            S1 = self.interpolantD[:,GP] / self.jacobian[i,GP]
            S00 = np.outer(S0, S0)
            S01 = np.outer(S0, S1)
            S10 = S01.T
            S11 = np.outer(S1, S1)
            # calculate transalations and rotations at Gauss point
            phiDGP = S1 @ phio # first order derivative of translation at Gauss point
            thetaGP = S0 @ theta # incremental rotation at Gauss point
            thetaDGP = S1 @ theta # first derivative of incremental rotation at Gauss point
            # calculate rotation matrices at Gauss point
            thetaGPMat = Rotation.from_rotvec(thetaGP).as_matrix() # incremental rotation matrix
            # nuGPMat = Rotation.from_rotvec(nuGP).as_matrix() # total rotation matrix
            nuGPMat = thetaGPMat @ self.rotationMatrixGP[i,GP,:, :]
            # calculate material properties at Gauss point
            matModF = nuGPMat @ self.inConstModForce[i] @ nuGPMat.T
            matModM = nuGPMat @ self.inConstModMoment[i] @ nuGPMat.T
            # calculate normal strain at Gauss point
            initialGamma = S1 @ self.initialCoordinates[self.connElem[i]] - nuGPMat @ self.E3 
            gammaGP = phiDGP - nuGPMat @ self.E3 - nuGPMat @ self.spuriousStrain[i, GP, :] # subtracting spurious strain due to mathematical definitino of geometry.
            # calculate force at Gauss point
            intForceGP = matModF @ gammaGP # force due internal stresses at Gauss point
            # calculate curvature strain at Gauss point: sum of term dependent on tangent operator and curvature strain at previous configuration
            zeta = thetaGPMat @ self.prevCurvatureStrain[i,GP,:] # term dependent on previous curvature strain
            tangentTheta = tangentCurvatureStrain(thetaGP) # tangent operator
            beta = tangentTheta @ thetaDGP # term dependent on the tangent operator of rotation tensor
            curvatureStrain = beta + zeta #+ nuGPMat @ omega0 @ nuGPMat.T
            if not checkRun:
                self.rotationMatrixGP[i,GP,:, :] = nuGPMat
                self.curvatureStrain[i,GP,:] = beta + zeta
            # calculate moment at Gauss point
            intMomentGP = matModM @ curvatureStrain # moment due internal stresses at Gauss point
            # calculate distributed loads at Gauss point
            distForceGP = S0 @ self.distributedForce[i, :, :]
            distMomentGP = S0 @ self.distributedMoment[i, :, :]
            if self.follower:
                followerDistForceGP = S0 @ self.followerDistForce[i, :, :]
                rotatedFollDistForceGP = nuGPMat @ followerDistForceGP
            # compute stiffenss equation
            for a in range(self.DOFPN//2):
                self.intResidualElemPDOF[f"R{a}"][:] += -intForceGP[a] * S1 * constant
                self.intResidualElemPDOF[f"R{a+3}"][:] += ((-intMomentGP[a]) * S1 \
                                                        + (intForceGP[:] @ self.perm[:,a,:] @ phiDGP[:]) * S0) * constant
                self.extResidualElemPDOF[f"R{a}"][:] += distForceGP[a] * S0 * constant
                if self.follower:
                    self.extResidualElemPDOF[f"R{a}"][:] += rotatedFollDistForceGP[a] * S0 * constant # type: ignore
                self.extResidualElemPDOF[f"R{a+3}"][:] += distMomentGP[a] * S0 * constant
                for b in range(self.DOFPN//2):
                    self.intTangentElemPDOF[f"T{a}{b}"][:,:] += matModF[a,b] * S11 * constant
                    self.intTangentElemPDOF[f"T{a}{b+3}"][:,:] += (phiDGP[:] @ self.perm[:,b,:] @ matModF[a,:] - intForceGP[:] @ self.perm[:,b,a]) * S10 * constant
                    if self.follower:
                        self.extTangentElemPDOF[f"T{a}{b+3}"][:,:] += (self.perm[a, :, b] @ rotatedFollDistForceGP[:]) * S00 * constant # type: ignore
                    self.intTangentElemPDOF[f"T{a+3}{b}"][:,:] += ( phiDGP[:] @ self.perm[:,a,:] @ matModF[:,b] + intForceGP[:] @ self.perm[:,b,a]) * S01 * constant
                    self.intTangentElemPDOF[f"T{a+3}{b+3}"][:,:] += ((-(phiDGP[:] @ self.perm[:,a,:] @ matModF[:,:] @ self.perm[:,b,:] @ phiDGP[:]) + (phiDGP[:] @ self.perm[a,:,:] @ (intForceGP[:] @ self.perm[:,b,:]))) * S00 \
                                                            + (-intMomentGP[:] @ self.perm[:,b,a]) * S10 \
                                                            + matModM[a,b] * S11 ) * constant

        # assemble external tangent element matrix because of follower force
        if self.follower:
            self.followerVector[:] = applyFollowerNBC(i, 1.0, self.rotation, self.connElem, self.followerVector, self.followerNBC)
            for a in range(self.DOFPN//2):
                for b in range(self.DOFPN//2):
                    for nnpel in range(self.NNPEL):
                        dof = nnpel * self.DOFPN
                        index = [0 + dof, 1 + dof, 2 + dof]
                        self.extTangentElemPDOF[f"T{a}{b+3}"][nnpel, nnpel] += self.perm[a, :, b] @ self.followerVector[index]
            

        # assemble element level matrices
        for ii in range(self.DOFPN):
            self.intResidualElem[ii::self.DOFPN] = self.intResidualElemPDOF[f"R{ii}"]
            self.extResidualElem[ii::self.DOFPN] = self.extResidualElemPDOF[f"R{ii}"]
            for jj in range(self.DOFPN):
                self.intTangentElem[ii::self.DOFPN, jj::self.DOFPN] = self.intTangentElemPDOF[f"T{ii}{jj}"]
                self.extTangentElem[ii::self.DOFPN, jj::self.DOFPN] = self.extTangentElemPDOF[f"T{ii}{jj}"]
            
    def assemble(self, loadratio:float, previousSolution:np.ndarray, incrementalSolution:np.ndarray, checkRun=False, initialStep=False):
        '''
        This function calls element() to calculate element level stiffness equation, applied boundary condition and assembles the global stiffness equation in the sparse form.
        Arguments:
            loadratio: float, %age of load to analyse in this step.
            previousSolution: np.ndarray of shape total nodes x DOFPN. It is passed to the element(). 
            incrementalSolution: np.ndarray of shape total nodes x DOFPN. It is passed to the element().
            checkRun: boolean, identify if the new assemble call is for calculating new state or analysing previous state.
            initialStep: boolean, check if first iteration of the model.
        Returns:
            tangentGlobal: np.ndarray, global tangent stiffness matrix
            residualGlobal: np.ndarray, global residual vector
            extResidualGlobal: np.ndarray, global external force vector, remains unchanged. Code can be modified to compute this once.
        '''

        # memory allocation
        extResidualGlobal = np.zeros(shape=self.shapeGM, dtype=np.float64)
        residualGlobal = np.zeros(shape=self.shapeGM, dtype=np.float64)
        tangent = np.zeros(shape=[self.NEL * self.elemComponents], dtype=np.float64)

        for i in range(self.NEL):
            self.residualElem[:] = 0.0 # rezero
            self.tangentElem[:,:] = 0.0
            self.element(i, previousSolution, incrementalSolution, checkRun) # call element() to compute element level numbers
            self.residualElem[:] = self.intResidualElem + loadratio * self.extResidualElem
            self.tangentElem = self.intTangentElem + loadratio * self.extTangentElem
            # apply boundary conditions to external force vector
            _, self.extResidualElem[:] = applyEBC(i, self.tangentElem, self.extResidualElem, self.EBC, initialStep)
            self.extResidualElem[:] = applyNBC(i, 1.0, self.extResidualElem, self.NBC) # loadratio = 1.0 to preserve total external force vector
            # apply boundary conditions to element equations - this segment is for calculating solution
            self.tangentElem[:,:], self.residualElem[:] = applyEBC(i, self.tangentElem, self.residualElem, self.EBC, initialStep)
            self.residualElem[:] = applyNBC(i, loadratio, self.residualElem, self.NBC)
            if self.follower:
                self.extResidualElem[:] = applyFollowerNBC(i, 1.0, self.rotation, self.connElem, self.extResidualElem, self.followerNBC)
                self.residualElem[:] = applyFollowerNBC(i, loadratio, self.rotation, self.connElem, self.residualElem, self.followerNBC)
            # assemble global system
            extResidualGlobal[self.connDOF[i]] += self.extResidualElem
            residualGlobal[self.connDOF[i]] += self.residualElem
            tangent[i * self.elemComponents : (i+1) * self.elemComponents] = self.tangentElem.flatten()

        tangentGlobal = csr_matrix((tangent, (self.sparRow, self.sparCol)), shape=(self.shapeGM, self.shapeGM), dtype=np.float64)
        return tangentGlobal, residualGlobal, extResidualGlobal

    def solveMatrixEquation(self, tangentGlobal:csr_matrix, residualGlobal:np.ndarray):
        '''
        This function solves the given K and F matrices.
        Return
            delSoln: np.ndarray of size shapeGM. Global incremental solution vector
        '''
        delSoln = np.zeros(shape=self.shapeGM, dtype=np.float64)
        delSoln[:] = np.asarray(spsolve(tangentGlobal, residualGlobal))

        return delSoln

    def configurationUpdate(self, delSoln:np.ndarray):
        '''
        This function adds the incremental solution to previous state of the beam.
        Arguments:
            delSoln: np.ndarray, output of solveMatrixEquation(). Contains incremental solution at all degrees of freedom at all nodes.
            initialStep: boolean. Check if it is the first iteration of the model.
        Returns:
            None. Writes predefined arrays.
        '''
        self.delConfig[:,:] = 0.0
        # self.prevConfig[:,:] = self.newConfig if not initialStep else 0.0
        self.newConfig[:] = 0.0
        # modifying incremental solution vector
        for ii in range(self.DOFPN):
            self.delConfig[:,ii] = delSoln[ii::self.DOFPN]

        # updating translations
        self.newConfig[:,0] = self.prevConfig[:,0] + self.delConfig[:,0] # translaiton along E1
        self.newConfig[:,1] = self.prevConfig[:,1] + self.delConfig[:,1] # translation along E2
        self.newConfig[:,2] = self.prevConfig[:,2] + self.delConfig[:,2] # translation along E3

        # updating rotations
        
        for node in range(self.totalNodes):
            # calculate rotation tensor at previous configuration
            prevRotationVec = self.prevConfig[node,[3,4,5]]
            prevRotationTensor = Rotation.from_rotvec(prevRotationVec).as_matrix()
            # calculate incremental rotation tensor
            incRotationVec = self.delConfig[node, [3,4,5]]
            incRotationTensor = Rotation.from_rotvec(incRotationVec).as_matrix()
            # calculate total rotation tensor
            newRotationTensor = incRotationTensor @ prevRotationTensor
            newRotationVec = Rotation.from_matrix(newRotationTensor).as_rotvec()
            # update
            self.newConfig[node,[3,4,5]] = newRotationVec

        # update arrays used in element()
        self.translation[:,:] = self.newConfig[:,[0,1,2]]
        self.rotation[:,:] = self.newConfig[:,[3,4,5]]
        self.incRotation[:,:] = self.delConfig[:, [3,4,5]]
        self.prevConfig[:,:] = self.newConfig

    def generalisedForce(self, solution:np.ndarray, incSolution:np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        This function computes the forces and moment in the beam due to the given deformation in the beam.
        Argument:
            solution: np.ndarray of shape totalNodes x DOFPN. It contains the configuration of the beam as given by the six degrees of freedom at every node at current state
            incSolution: np.ndarray of shape totalNodes x DOFPN. It contains the incremental configuration of the beam as given by the six degrees of freedom at every node at current state.
        Returns:
            forces: np.ndarray of shape NEL x NGQP x 3. Forces along E1, E2 and E3 axis.
            moments: np.ndarray of shape NEL x NGQP x 3. Moments about E1, E2 and E3 axis.
            z: np.ndarray of shape NEL x NGQP. Coordinate of Gauss point along the beam (E3 axis).
        '''
        forces = np.zeros(shape=[self.NEL, self.NGQP, 3], dtype=np.float64)
        moments = np.zeros(shape=[self.NEL, self.NGQP, 3], dtype=np.float64)
        z = np.zeros(shape=[self.NEL, self.NGQP], dtype=np.float64)

        for i in range(self.NEL):
            phio = solution[self.connElem[i],[0,1,2]]
            nuo = solution[self.connElem[i],[3,4,5]]
            theta = incSolution[self.connElem[i],[3,4,5]]
            prevCurvatureStrain = self.curvatureStrain[i,:,:]
            for GP in range(self.NGQP):
                S0 = self.interpolant[:,GP]
                S1 = self.interpolantD[:,GP] / self.jacobian[i,GP]
                # calculate transalations and rotations at Gauss point
                z[i, GP] = S0 @ self.globalCoords[i]
                phiDGP = S1 @ phio # first order derivative of translation at Gauss point
                nuGP = S0 @ nuo # total rotation at Gauss point
                thetaGP = S0 @ theta # incremental rotation at Gauss point
                thetaDGP = S1 @ theta # first derivative of incremental rotation at Gauss point
                # calculate rotation matrices at Gauss point
                thetaGPMat = Rotation.from_rotvec(thetaGP).as_matrix() # incremental rotation matrix
                nuGPMat = Rotation.from_rotvec(nuGP).as_matrix() # total rotation matrix
                # calculate material properties at Gauss point
                matModF = nuGPMat @ self.inConstModForce[i] @ nuGPMat.T
                matModM = nuGPMat @ self.inConstModMoment[i] @ nuGPMat.T
                # calculate normal strain at Gauss point
                gammaGP = phiDGP - nuGPMat @ self.E3
                # calculate force at Gauss point
                intForceGP = matModF @ gammaGP # force due internal stresses at Gauss point
                # calculate curvature strain at Gauss point: sum of term dependent on tangent operator and curvature strain at previous configuration
                zeta = thetaGPMat @ prevCurvatureStrain[GP,:] # term dependent on previous curvature strain
                normThetaGP = np.linalg.norm(thetaGP)
                dummy = np.sin(normThetaGP) / normThetaGP
                beta = dummy * thetaDGP \
                    + (1 - dummy) * (thetaGP @ thetaDGP / normThetaGP) * thetaGP / normThetaGP \
                    + (2 * np.sin(0.5*normThetaGP)**2 / normThetaGP**2) * np.cross(thetaGP, thetaDGP) if normThetaGP > 1E-15 else thetaDGP # term dependent on the tangent operator of rotation tensor
                curvatureStrain = beta + zeta
                # calculate moment at Gauss point
                intMomentGP = matModM @ curvatureStrain # moment due internal stresses at Gauss point
                # allocate
                forces[i,GP,:] = intForceGP
                moments[i,GP,:] = intMomentGP

        return forces, moments, z

    def computeDiagnostics(self, solution:np.ndarray, incSolution:np.ndarray, loadratio:float, check, iteration, convergence:bool):
        '''
        This function computes diagnostic parameters for determinig the status of the nonlinear solver.
        Argument:
            solution: np.ndarray of shape totalNodes x DOFPN. It contains the configuration of the beam as given by the six degrees of freedom at every node at current state
            incSolution: np.ndarray of shape totalNodes x DOFPN. It contains the incremental change in state of the beam as given by the six degrees of freedom at every node.
            loadratio: float, %age of total load applied in the current step
            check: float, convegence check
            iteration: int, current iteration number after convergence
            convergence: boolean, True if this funciton is called after convergence' else False. Identifies the right key to write data in.
        Returns:
            None. Writes predefined dictionary.
        '''
        result = 'converged' if convergence else 'diverged'
        #store
        self.diagnostics[result]['loadratioHistory'].append(loadratio)
        self.diagnostics[result]['iteration'].append(iteration+1)
        self.diagnostics[result]['solution'].append(solution.copy())
        self.diagnostics[result]['incSolution'].append(incSolution.copy())
        self.diagnostics[result]['error'].append(check)

        # compute current state of tangent and residual
        newTangent, newResidual, newExtForce = self.assemble(loadratio, solution, incSolution, True)
        self.diagnostics[result]['residualNormHistory'].append(np.linalg.norm(newResidual))

        # check for symmetry if tangent matrix
        tangentDense = newTangent.toarray()
        tangentNorm = max(np.linalg.norm(tangentDense), 1.0)
        symmetryError = (np.linalg.norm(tangentDense - tangentDense.T)/ tangentNorm)
        self.diagnostics[result]['tanSymmHistory'].append(symmetryError)
        # check for numberical condition of tangent matrix
        singularValues = np.linalg.svd(tangentDense, compute_uv=False)
        sigmaMax = singularValues[0]  
        sigmaMin = singularValues[-1]
        self.diagnostics[result]['singHistory'][0].append(sigmaMin)
        self.diagnostics[result]['singHistory'][1].append(sigmaMax)
        if sigmaMin > 0.0:
            self.diagnostics[result]['condNumHistory'].append((sigmaMax / sigmaMin))
        else:
            self.diagnostics[result]['condNumHistory'].append(np.inf)
        # compute minimum eigen value of the system
        tangentEigenvalues = np.linalg.eigvalsh(tangentDense)
        self.diagnostics[result]['stabilityEigenvalueHistory'].append(tangentEigenvalues[0])
        # compute eigenvalues of free DOFs and determine the transition from positive to negative (loss of stability)
        tanDense = tangentDense[np.ix_(self.freeDOFs, self.freeDOFs)]
        value, Vector = np.linalg.eig(tanDense)
        valueSort = np.sort(value[np.abs(value.imag) <= 1e-10 * np.abs(value).max()].real) # type: ignore
        self.diagnostics[result]['morseHistory'].append(np.sum(valueSort < 0)) # identify the number of negative eigenvalues
        self.diagnostics[result]['minEigValueHistory'].append([valueSort[0], valueSort[1]].copy()) # minimum eigen value
        
        workdone = Vector[:,0] @ newExtForce[self.freeDOFs] / max(np.linalg.norm(newExtForce[self.freeDOFs]), 1e-30) # normalise the external force vector and dot product with the net displacement vector - should give the net work done.
        self.diagnostics[result]['workdone'].append(workdone)  
        if self.solverType == 'Load-controlled':
            if result == 'diverged' or valueSort[0] < 0.0 : # check for divergence or loss of stability( minimum eigen value < 0)
                self.solverType = 'Arc-length'  
                print(f'Switching to Arc-length solver at loadratio {loadratio}: last result = {result}, minimum eigenvalue = {valueSort[0]}')
                
    def newtonRaphson(self, loadratio, loadstep):
        '''
        This function is the the solution method for the load controlled solver.
        '''
        convergence = False
        totaliterations = 0
        check = 0.0
        recall = (self.rotationMatrixGP.copy(), self.curvatureStrain.copy()) # remember gauss point rotations and curvatures if NR diverges.
        for iterationNR in range(self.loopIterations):
            initialStep = True if loadstep == 0 and iterationNR == 0 else False
            tangent, residual, _ = self.assemble(loadratio, self.prevConfig, self.delConfig, False, initialStep)
            delsoln = self.solveMatrixEquation(tangent, residual)
            self.configurationUpdate(delsoln)
            check = convergenceCheck(delsoln, self.newConfig.flatten())
            if check < self.convergenceCriteria:
                convergence = True
                totaliterations = iterationNR
                break
            elif iterationNR + 1 == self.loopIterations:
                convergence = False
                print(f'Model did not converge at loadratio {loadratio}.')
            totaliterations = iterationNR
        if not convergence:
            self.rotationMatrixGP[:], self.curvatureStrain[:] = recall
        self.computeDiagnostics(self.newConfig, self.delConfig, loadratio, check, totaliterations, convergence)
        return convergence

    def solverFEM(self):
        '''
        Solver function for the finite element model.
        '''
        convergence = []
        iteration, loadratio = 0, 0.0 # initilise forward path
        loadratio = 0.0
        loadstep = 0.0
        while loadratio < 1.0 and iteration < self.maxIterations :
            if iteration % 200 == 0:
                print(f'Iteration: {iteration}, Loadratio: {loadratio}')
            if self.solverType == 'Load-controlled':
                loadratio = (loadstep + 1) / self.loadsteps
                convergence.append(self.newtonRaphson(loadratio, loadstep))
            else:
                if self.arclengthClass is None:
                    self.arclengthClass = Arclength(self)
                self.arclengthClass.arcLengthSolver(iteration)
                if self.arclengthClass.stopALSolver:
                    print(f'Arc length solver failed to converge at iteration {iteration} and loadratio {self.arclengthClass.totalLoadRatio}')
                    break
                self.newConfig[:] = transformDimension(self.arclengthClass.ultimateConvergedSoln)
                self.delConfig[:] = transformDimension(self.arclengthClass.ultimateIncConvergedSoln)
                loadratio = self.arclengthClass.totalLoadRatio
                alIterations = self.arclengthClass.countALIteration
                check = [self.arclengthClass.check1, self.arclengthClass.check2]
                self.computeDiagnostics(self.newConfig, self.delConfig, loadratio, check, alIterations, self.arclengthClass.solutionOK) # type: ignore
            iteration += 1
            loadstep += 1
        return convergence

        

        



        


        


        
        








        

