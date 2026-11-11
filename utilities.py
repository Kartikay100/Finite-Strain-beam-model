'''
Author:      Kartikay Shukla
File:        utilities.py
Created:     October 27, 2026
LM:          October 28, 2026

Description
This file contains function for different mathematical operations in the finite element model.
'''

import numpy as np
import scipy as sc
import json
from datetime import datetime
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def inputJson():
    '''
    Function to read input from JSON file.
    '''
    with open('input/inputJSON.json', mode='r', encoding='utf-8') as read_file:
        inputs = json.load(read_file)

    modelingConditions = inputs[0]
    geometry = inputs[1]
    materialProperty = inputs[2]
    essentialBC = inputs[3]
    naturalBC = inputs[4]
    distributedForce = inputs[5]
    distributedMoment = inputs[6] 
    followerLoad = inputs[7]

    return modelingConditions, geometry, materialProperty, essentialBC, naturalBC, distributedForce, distributedMoment, followerLoad

def timenow():
    '''
    Function to return current time stamp in YYYYMMDD-HHMMSS
    '''
    now = datetime.now() # get current time
    return now.strftime('%Y%M%D-%H%M%S')

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# Numerical interpolation and integration functions -------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

def glQuad(NGQP:int)-> tuple[np.ndarray, np.ndarray]:
    '''
    This function returns Gauss Legendre quadrature points for given number of points, which is used for numerical integration.
    Arguments:
        NGQP: Number of Gauss quadrature points
    Returns:
        gqP: Gauss quadrature points, numpy.float64
        gqW: Gauss quadrature weights, numpy.float64
    '''

    gqP, gqW = np.polynomial.legendre.leggauss(NGQP) # call gauss points and weights from leggauss()

    return gqP.astype(np.float64), gqW.astype(np.float64)

def spectralNodes(n:int)-> np.ndarray:
    '''
    This function creates an array of spectral nodal points (also called as Gauss-Lobatto-Legendre points) over the interval [-1,1].
    Gauss-Lobatto-Legendre points also include the end points (-1 and 1).
    It is useful for defininig nodal location in an element. The order of polynomial is defined by the number of nodes - 1.
    Minimum value for n is 2, which corresponds to a linear element. For n=1, the function returns a single node at 0.
        n: number of nodes. Defines the degree of polynomial.
    Returns:
        specNodes: array of spectral nodes, numpy.ndarray
    '''
    if n<2:
        raise RuntimeError(f"n must be greater than or equal 2, got {n}.")

    polynomialOrder = n - 1
    # first derivative of legendre polynomials
    d1Poly = sc.special.legendre(polynomialOrder).deriv() # calling polyOrder-th order legendre polynomial and taking its derivative
    # calculating roots of the derivative of legendre polynomial to get the interior nodes
    roots = np.sort(d1Poly.roots) # roots are sorted in ascending order
    # create an array of spectral nodes
    specNodes = np.insert(roots, [0, polynomialOrder-1], [-1,1]) # insert -1 and 1 at the beginning and end of the array of roots. polyOrder used because for 2 nodes there are 0 roots.

    return specNodes.astype(np.float64)

def lagrangePoly(NNPEL:int, NGQP:int)-> tuple[np.ndarray, np.ndarray]:
    '''
    This function creates an array of coefficients of Lagrange polynomials (and their first derivatives) to interpolate the values from nodal (spectral) points to Gauss-Legendre points.
    This interpolation will help with numerical integration of the element stiffness matrix and force vector, in a one-dimensional analysis.
    More information about Lagrange polynomials can be found at https://en.wikipedia.org/wiki/Lagrange_polynomial.
    Array of values for polynomial is calculated to interpolate nodal location to Gauss-Legendre points.

    NNPEL: int,  Number of nodes per element.
    NGQP: int, Number of Gauss quadrature points.

    Returns:
        lagrangePoly: array of coefficients of Lagrange polynomials for interpolating from nodal location to GP location, numpy.ndarray
        lagrangePolyDerv1: array of coefficients of first derivative of Lagrange polynomials for interpolating from nodal location to GP location, numpy.ndarray
    '''
    if NNPEL<2:
        raise RuntimeError(f"Number of nodes per element must be greater than or equal 2, got {NNPEL}.")

    # get local nodal coodinates in an element - array of spectral points on the element in [-1,1]
    
    domain = spectralNodes(NNPEL)
    # get Gauss Legendre points within the element, for numerical integration
    GP, _ = glQuad(NGQP) # points for numerical integration

    # memory allocation for the array of coefficients of Lagrange polynomials
    lagrangePoly = np.zeros((NNPEL, NGQP), dtype=np.float64)
    lagrangePolyDerv1 = np.zeros((NNPEL, NGQP), dtype=np.float64)

    for node1, nodeLoc1 in enumerate(domain):
        mask = np.arange(NNPEL) != node1 # create a boolean array with nodal indices hidden if they are equal to the current node index n1.
        denominator = nodeLoc1 - domain[mask] # xj - xi for i != j, in the Lagrange polynomial formula, vectorised
        phi = 1.0
        phiD1 = 0.0
        for gp in range(NGQP):
            constant = GP[gp] - domain[mask] # x - xi for i != j, numerator in the Lagrange polynomial formula, vectorised
            # compute Lagrange polynomial and its first derivative
            phi = np.prod(constant / denominator)
            phiD1 = np.sum ( 1.0 / constant)
            # store
            lagrangePoly[node1, gp] = phi
            lagrangePolyDerv1[node1, gp] = phiD1 * phi

    return lagrangePoly, lagrangePolyDerv1

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# Mesh functions ------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

def mesh1D(L:float, NEL:int, NNPEL:int, DOFPN=6)-> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    '''
    This function creates a one-dimensional mesh of the defined domain (length) in a finite element analysis.
    Nodal locations are defined based on the spectral points (Gauss-Lobatto-Legendre). It leads to non-uniform mesh with higher density of nodes at the ends of the domain.
    Arguments:
        L: float, length of each component. The sum of this list would give the total length of the beam.
        NEL: int, number of elements to be used to discretize each component.
        NNPEL: int, number of nodes per element. The degree of approximating polynomial is decided baased on NNPEL - 1. 
        DOFPN: int, number of degrees of freedom per node, 6 for Simo-Vu-Quoc beam model.
    Returns:
        connElem: array of connectivity between elements and nodal index in the domain, numpy.ndarray
        globalCoords: array of global coordinates of nodes in the domain, numpy.ndarray
        h: length of each element, numpy.float64
        connDOF: array of connectivity between elements and degree of freedom index in the domain, numpy.ndarray
        totals: list containing value of total number of elements, total number of nodes and equations per element.
    '''
    # check
    if NNPEL <= 1:
        raise ValueError("NNPEL must be positive integer greater than 1.")
    domain = spectralNodes(NNPEL)
    eqnsPerElem = NNPEL * DOFPN
    
    # memory allocation
    h = L / NEL
    connElem = np.zeros(shape=[NEL,NNPEL], dtype=np.int32) # connectivity between elements and nodal index in the domain.
    globalCoords = np.zeros(shape=[NEL,NNPEL], dtype=np.float64)
    connDOF = np.zeros(shape=[NEL, eqnsPerElem], dtype = np.int32) # connectivity between elements and degree of freedom index in the domain.

    for elem in range(NEL):
        connElem[elem, :] = np.arange(NNPEL) if elem == 0 else connElem[elem-1, :] + (NNPEL - 1)
        # global coordinates
        x0 = elem * h
        x1 = (elem+1) * h
        # linearly interpolate to find the sepctral points in global coordinate system within the element
        globalCoords[elem,:] = 0.5 * ( (1 - domain) * x0 + (1 + domain) * x1)
        for j in range(DOFPN):
            connDOF[elem, j:eqnsPerElem:DOFPN] = connElem[elem,:] * DOFPN + j

    return connElem, globalCoords, h, connDOF,

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# Boundary functions --------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

def boundaryHandler(inputBC:dict, connElem:np.ndarray, type:str, DOFPN=6):
    '''
    The boundary condition in the input file is a dictionary with keys as 'globalNodes#' and 'Values'. 
    'globalNodes#' is a list of list. [key][0] is a list of global node indices and [key][1] is a list of corresponding number of boundary conditions at that node. The nodes and number of boundary conditions can be correlated by using the same index number in both the lists. The sum of value in number of boundary conditions is the total number of boundary condition.
    'Values' is a list of list of size [n,2] where n is the total number of boundary conditions. [:,0] is a local degree of freedom number (local to the node) and [:,1] is the value of the boundary condition at that degree of freedom. The order of the values in the list is same as the order of the nodes in 'globalNodes#' and number of boundary conditions at that node in 'globalNodes#'.

    This function creates a dictionary with elements with boundary condition as keys. Each key has 3 lists, local node numberings of nodes with boundary conditions, DOF numberings local to element with boundary condition and the boundary value. The values in the three lists can be correlated with the same index number.

    This function digests the boundary condition dictionary and returns a dictionary with keys Element, localNode#, localDOF#, Values.
    Arguments:
        DOFPN: int, number of degrees of freedom per node.
        inputBC: dict, dictionary of boundary conditions from the input.
        connElem: np.ndarray, connectivity between elements and nodal index in the domain.
        type: string, type of boundary condition for which this function is called. Takes two values 'EBC' and 'NBC'.
    Returns:
        constraints: nested dictionary with elements with boundary condition as keys. 'localNode#', 'localDOF#', and 'Values' are keys of the dicitonary within each element. 
            Each key is a list and each related parameter can be accessed with the same index in every list. 
        This structure lets us access the element with boundary conditions quickly [O(1) search instead of O(N) search with np.where()].
    '''

    nodes = np.array(inputBC['globalNode#'][0])
    numOfValues = np.array(inputBC['globalNode#'][1], dtype=np.int32)
    values = np.array(inputBC['Values']) # multi row, two column array, [:0] contains the DOF # and [:1] contains corresponding boundary condition
    boundaries  = np.concatenate([[0], np.cumsum(numOfValues)]) # take cummulative sum of boundary for each node.

    constraints = {}

    for n, node in enumerate(nodes):
        index = np.where(connElem == node) # returns a tuple of indices with row index (element #) and column index (node #) for each 'node'
        elemNum, localNodeNum = index[0], index[1] # each of size (K,)
        M = numOfValues[n] # number of boundary conditions specified for this node
        K = len(elemNum) # number of elements that share this node
        sliceBC = values[boundaries[n]:boundaries[n+1],:] # size (M,2)
        localDOFs = [int(ii) for ii in sliceBC[:,0]]
        localDOF_elem = np.array([localDOF + localNodeNum * DOFPN for localDOF in localDOFs], dtype=np.int32).reshape(-1) # calculate DOFs index local to the element
        value = sliceBC[:,1] / K  if type == 'NBC' and K != 0.0 else sliceBC[:,1]# size M, divide the value by number of elements sharing the node to avoid double counting of the boundary condition.
        # create arrays
        elemNums_flat = np.tile(elemNum, M)
        localNodes_flat = np.tile(localNodeNum, M)
        values_flat = np.repeat(value, K)
        # store, total size = M + K
        for i, elem in enumerate(elemNums_flat):
            if elem not in constraints:
                constraints[elem] = {'localNode#': [], 'localDOF_elem#': [], 'localDOF_node#':[], 'Values':[]}
            constraints[elem]['localNode#'].append(localNodes_flat[i].tolist())
            constraints[elem]['localDOF_elem#'].append(localDOF_elem[i].tolist())
            constraints[elem]['Values'].append(values_flat[i])

    return constraints

def applyEBC(elemIndex:int, SME:np.ndarray, CVE:np.ndarray, boundaryCondition:dict, initialStep=False)-> tuple[np.ndarray, np.ndarray]:
    '''
    This function applies the essential boundary conditions to the element level stiffness matrix and force vector. 
    Essential BC is also called as Dirchlet BC.

    Arguments:
        elemIndex: int, index of the element for which SME and CVE has been calculated.
        SME: np.ndarray, element level stiffness matrix of size [NNPEL x DOFPN, NNPEL x DOFPN].
        CVE: np.array, element level force vector of size [NNPEL x DOFPN].
        boundaryCondition: dict, nested dictionary of boundary conditions keyed by element number, outputof boundaryHandler().
        initialStep: Boolean. True/False. True if iteration# = 0 and load step # = 0 i.e. first iteration of the simulation. Otherwise, False.
    Returns
        SME, CVE: with boundary codition applied.
    '''
    
    if elemIndex in boundaryCondition: # check if element # is in the boundary condition list
        for index, localDOFIndex in enumerate(boundaryCondition[elemIndex]['localDOF_elem#']):
            boundaryValue = boundaryCondition[elemIndex]['Values'][index] if initialStep else 0.0
            # apply boundary condition while preserving the symmetricity of the SME
            coefficient = SME[localDOFIndex, localDOFIndex]
            # zero out the Node-DOF# row where boundary condition is applied
            SME[localDOFIndex, :] = 0.0
            # modify the Node-DOF# column where boundary condition is applied as
            CVE[:] = CVE - boundaryValue * SME [ :, localDOFIndex]
            SME[:, localDOFIndex] = 0.0 # zero out the Node-DOF# column where boundary condition, symmetry preserved
            # preserve the diagonal value in stiffness matrix, this preserves the order of the stiffness matrix.
            SME[localDOFIndex, localDOFIndex] = coefficient
            # modify the force vector value : with scaled boundary condition value
            CVE[localDOFIndex] = boundaryValue * coefficient

    return SME, CVE

def applyNBC(elemIndex:int, loadRatio:float, CVE:np.ndarray, boundaryCondition:dict)-> np.ndarray:
    '''
    This function applies the natural boundary conditions to the force vector. 
    Natural BC is also called as Neuman BC.

    Arguments:
        elemIndex: int, index of the element for which SME and CVE has been calculated.
        loadRatio: np.float64, ratio of the total laod that needs to be applied in this iteration.
        CVE: np.array, element level force vector of size [NNPEL x DOFPN].
        boundaryCondition: dict, nested dictionary of boundary conditions keyed by element number, outputof boundaryHandler().
    Returns
        CVE: with boundary codition applied.
    '''
    if elemIndex in boundaryCondition:
        for index, localDOFIndex in enumerate(boundaryCondition[elemIndex]['localDOF_elem#']):
            CVE[localDOFIndex] += boundaryCondition[elemIndex]['Values'][index] * loadRatio
    return CVE

def applyFollowerNBC(elemIndex:int, loadRatio:float, totalRotationArray:np.ndarray, connElem:np.ndarray, CVE:np.ndarray, boundaryCondition:dict)-> np.ndarray:
    '''
    This function applies the follower natural boundary conditions to the force vector. 
    It only works for follower load and not follower moments.
    Natural BC is also called as Neuman BC.

    Arguments:
        elemIndex: int, index of the element for which SME and CVE has been calculated.
        loadRatio: np.float64, ratio of the total laod that needs to be applied in this iteration.
        totalRotationArray: np.ndarray of rotation vector at global nodes.
        connElem: np.ndarray, node-element connectivity array.
        CVE: np.array, element level force vector of size [NNPEL x DOFPN].
        boundaryCondition: dict, nested dictionary of boundary conditions keyed by element number, outputof boundaryHandler().
    Returns
        CVE: with boundary codition applied.
        dummyCVE: boundary codition.
    '''
    DOFPN = 6
    _, NNPEL = connElem.shape
    if elemIndex in boundaryCondition:
        dummyforceVector = np.zeros(shape=[NNPEL, 3], dtype=np.float64)        # memory allocation
        for index, localDOFIndex in enumerate(boundaryCondition[elemIndex]['localDOF_elem#']):
            nnpel = localDOFIndex // DOFPN # determine current node number
            dummyforceVector[nnpel, localDOFIndex % DOFPN] = boundaryCondition[elemIndex]['Values'][index] * loadRatio
        for nnpel in range(NNPEL):
            globalnode = connElem[elemIndex, nnpel]
            rotation = Rotation.from_rotvec(totalRotationArray[int(globalnode)]).as_matrix()
            dof = int(nnpel * DOFPN) # current DOF #
            dofpoints = [0 + dof, 1 + dof, 2 + dof]
            CVE[dofpoints] += rotation @ dummyforceVector [nnpel]
    return CVE

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# Other preprocessing functions --------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

def propertyPerElement(NEL:int, properties:list)-> list:
    '''
    This function builds arrays to define properties for every element.
    Argument
        NEL: int, number of elements
        properties: list of properties for which per element data structure is to be calculated.
    '''

    perElem = []
    for property in properties:
        propPerElement = np.zeros(shape=NEL, dtype=np.float64)
        for element in range(NEL):
            propPerElement[element] = property
        perElem.append(propPerElement)

    return perElem

def gendistributedForce(inputLoad:dict, connElem:np.ndarray, DOFPN:int=3)->np.ndarray:
    '''
    This function takes the input distributed (generalised) load and creates an array of NEL x NNPEL x 3 defining values at each node.
    Argument
        connElem: np.ndarray, connectivity between elements and nodal index in the domain.
        inputLoad: dictionary, it has a similar structure as inputBC in boundaryHandler()
        DOFPN: int, 3, used to size the array
    Return
        generalisedLoad: np.ndarray of shape NEL x NNPEL x 3 containing values of distributed load. 
            3 DOFPN chosen - this means that its either an array for distributed force (in three directions) or distributed moment. Hence 3 DOFPN.
    '''
    rows, columns = connElem.shape # rows = total number of elements, columns = number of nodes per element
    generalisedLoad = np.zeros(shape=[rows, columns, DOFPN], dtype=np.float64)
    
    nodes = np.array(inputLoad['globalNode#'][0])
    numOfValues = np.array(inputLoad['globalNode#'][1], dtype=np.int32)
    values = np.array(inputLoad['Values']) # multi row, two column array, [:0] contains the DOF # and [:1] contains corresponding boundary condition
    loads  = np.concatenate([[0], np.cumsum(numOfValues)]) # take cummulative sum of boundary for each node.

    for n, node in enumerate(nodes):
        elemNum, localNodeNum = np.where(connElem == node) # returns a tuple of indices with row index (element #) and column index (node #) for each 'node'
        K = len(elemNum) # number of elements that share this node
        sliceLoad = values[loads[n]:loads[n+1], :] # size (M,2)
        localDOFs = [int(ii) for ii in sliceLoad[:,0]]
        value = sliceLoad[:,1]  # size M, 
        # allocate
        for i, num in enumerate(elemNum):
            for j, dof in enumerate(localDOFs):
                generalisedLoad[num, localNodeNum[i], dof] = value[j]

    return generalisedLoad

def constitutiveMatrix(NEL:int, E:np.ndarray, G:np.ndarray, A:list, I:list)-> tuple[np.ndarray, np.ndarray]:
    '''
    This function builds a 3x3 matrix per element for constitutive assumptions for relating generalised force with generalised displacement.
    It returns two sets of NEL x 3 x 3 arrays, one relating force vs normal displacement and other relating moment with shear displacement.
    Argument:
        NEL: int, number of elements
        E: np.ndarray of Young's modulus, of shape NEL
        G: np.ndarray of shear modulus, of shape NEL
        A: list, [A1, A2, A3] which is area along E1, E2 and E2 axis per element. A1, A2, A3 are of shape NEL.
        I: list, [I1, I2, J] which is second moment of area along E1, E2 and E2 axis per element. I1, I2, J are of shape NEL.
    Return:
        matMatrixForce: np.ndarray of shape NEL x 3 x 3
        matMatrixMoment: np.ndarray of shape NEL x 3 x 3
    '''
    matMatrixForce = np.zeros(shape=[NEL, 3, 3], dtype=np.float64)
    matMatrixMoment = np.zeros(shape=[NEL, 3, 3], dtype=np.float64)

    for elem in range(NEL):
        force_normal = [G[elem]*A[0][elem], G[elem]*A[1][elem] , E[elem]*A[2][elem]]
        moment_shear = [E[elem]*I[0][elem], E[elem]*I[1][elem] , G[elem]*I[2][elem]]
        matMatrixForce[elem, :, :] = np.diag(force_normal)
        matMatrixMoment[elem, :, :] = np.diag(moment_shear)

    return matMatrixForce, matMatrixMoment

def initialCurvature(globalCoordinates:np.ndarray, initialCurvature:dict)->np.ndarray:
    '''
    This function computes an array of rotation vector for each global nodal point. It uses the global coordinates array, scales the give total rotation vector appropritaely and calculates the rotation vector at each nodal point.
    This function only works if rotation of all the points is about the same rotation vector (2D rotation), changing only by the rotation angle.
    Argument:
        globalCoordinates: np.ndarray, array of global coordinates of nodes of shape number of elements x number of nodes per element.
        initialCurvature: nested dict, with globalCoordinates as outer keys. Each outer key has two inner keys: rotationAngle and rotationVector. Coordinates in need to be arange in increasing order.
            globalCoordinates: dict, with keys rotationAngle, rotationVector.
                rotationAngle: float, total rotation angle from initial configuration.
                rotationVector: list of float, length = 3.
    Return:
        initialRotationVector: np.ndarray of shape totalNodes x 3. 
    '''

    coordinates = np.unique(globalCoordinates)
    totalNodes = len(coordinates)
    initialRotationVector = np.zeros(shape=[totalNodes, 3], dtype=np.float64) # memory allocation
    outerKeys = list(initialCurvature.keys()) # get keys
    for i, point in enumerate(coordinates):
        # initialise few parameters
        diffA = np.inf
        diffB = np.inf
        above = None
        below = None
        for j, outerKey in enumerate(outerKeys):
            # find outer keys which is right above or below 'point'
            if outerKey > point:
                diffAbove = outerKey - point
                if diffAbove < diffA:
                    above = outerKey
                    diffA = diffAbove
            elif outerKey < point:
                diffBelow = point - outerKey
                if diffBelow < diffB:
                    below = outerKey 
                    diffB = diffBelow                
            else:
                above = outerKey
                below = outerKey
        netRotation = initialCurvature[above]['rotationAngle'] - initialCurvature[below]['rotationAngle'] if above != below else initialCurvature[below]['rotationAngle']
        scale = ((point - below) / (above - below) )  if (above != below) else 1.0  # type: ignore
        scaledRotation = (initialCurvature[below]['rotationAngle'] if above != below else 0.0) + scale * netRotation 
        initialRotationVector[i,:] = scaledRotation * np.array(initialCurvature[above]['rotationVector'])
    return initialRotationVector

def initialPosition(globalCoordinates:np.ndarray, inputCurvature:np.ndarray, inputRotations:dict)->np.ndarray:
    '''
    This function calculates the initial coordinates of the nodes of the beam based on the initial rotations given by the initial curvature
    This function only works for planar rotations, may work for non planar rotations.
    The computation assumes a new 2 noded element, existing between consecutive nodes in flattened globalCoordinates. Hence newNNPEL is hardcoded as 2 and uses 5 gauss points for integration. 
    Arguments:
        globalCoordinates: np.ndarray, NEL x NNPEL, global coordinates of the nodes of the beam.
        inputCurvature: np.ndarray, totalNodes x 3, rotation vector at global nodal position.
        inputRotations: dict, rotations defined in the input. It will be useful when calling initialCurvature().
    Returns
        curvedPosition: np.ndarray, totalNodes x 3, initial coordinates of the nodes of the beam.
    '''
    newNNPEL = 2
    newNGQP = 10
    totalNodes, _ = inputCurvature.shape
    coordinates = np.unique(globalCoordinates)
    curvedPosition = np.zeros(shape=[totalNodes, 3], dtype=np.float64)
    E3 = np.array([0,0,1.0])

    gp, gw = glQuad(newNGQP) # Gauss Legendre points and weights
    lagPoly, _ = lagrangePoly(newNNPEL, newNGQP)
    relRotationVec = np.zeros(shape=[newNNPEL, 3], dtype=np.float64)
    deltaPosition = np.zeros(shape=3, dtype=np.float64)
    for node, coordinate in enumerate(coordinates):
        if node == 0:
            continue
        previousPtRot, currentPtRot = inputCurvature[node - 1], inputCurvature[node]
        prevPtRotMat = Rotation.from_rotvec(previousPtRot).as_matrix()
        currPtRotMat = Rotation.from_rotvec(currentPtRot).as_matrix()
        relRotationMat = prevPtRotMat.T @ currPtRotMat
        relRotationVec[1,:] = Rotation.from_matrix(relRotationMat).as_rotvec()
        deltaPosition[:] = 0.0
        h = coordinate - coordinates[node-1]
        gpCoordinates = np.array([coordinates[node-1], coordinate]) @ lagPoly # get coordinates of Gauss legendre points
        rotationsGLPoints = initialCurvature(gpCoordinates, inputRotations)# get rotation vectors at Gauss legendre points for numerical integration, array of size, GL points x 3
        for GP in range(newNGQP): 
            constant = gw[GP] * h / 2
            rotMatGP = Rotation.from_rotvec(rotationsGLPoints[GP,:]).as_matrix()
            deltaPosition[:] += constant * (rotMatGP @ E3)
        curvedPosition[node, :] = curvedPosition[node - 1, :] + deltaPosition

    return curvedPosition

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# Miscellaneous functions ---------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

def permutationOp()->np.ndarray:
    '''
    This function returns a 3x3x3 array of values of permutation operator
    Returns
        epsilon: np.ndarray
    '''
    i, j, k = np.indices((3,3,3))
    Levi_Civita = 0.5 * (i-j) * (j-k) * (k-i)
    return Levi_Civita

def convergenceCheck(vector1, vector2, formula=None):
    '''
    This function has formula for defining the convergence criteria of a Finite element model, based on the input vectors.
    Argument:
        vector1: np.ndarray, can be change in solution
        vector2: np.ndarray, can be total solution
        case: place folder argument. Can include other comparative measures and call them as needed using this argument.
    Return:
        check: np.float64
    '''

    check = 0.0
    # Root mean squared error - this check calculates the ratio of RMSE error of two (given) vectors.
    numer = np.sum(vector1 ** 2)
    denom = np.sum(vector2 ** 2)
    if denom > 1E-16:
        check = np.sqrt( numer / denom)
    else:
        raise ValueError('Vector 2 is a zero array. Cannot calculate RMSE ratio. Define or use another criteria.')

    return check

def saveData(solver, fileName):
    '''Saves diagnostic information of the run'''
    inputToModel = {'modelingCond': solver.modelingConditions, 'geometry': solver.geometry, 'props': solver.materialProperty, \
                'essentialBC': solver.essentialBC, 'naturalBC': solver.naturalBC, \
                'distributedForce': solver.distributedForce, 'distributedMoment': solver.distributedMoment, \
                'follower': solver.follower
            }
    np.savez(f'data/{fileName}', inputToModel=np.array(inputToModel, dtype=object), diagnostics=np.array(solver.diagnostics, dtype=object))

def tangentCurvatureStrain(v:np.ndarray)->np.ndarray:
    '''
    This function computes the tangent term of the given nodal rotation vector. 
    Created a separate function as it is a complicated formula and its easier to call this function.
    Arguments:
        v: np.ndarray, size (3), v has already been interpolated to Gauss point.
    '''

    normV = np.linalg.norm(v)

    Identity = np.eye(3)

    if normV < 1E-15:
        return Identity
    else:
        term1 = (1 - np.cos(normV)) / normV**2
        term2 = (1 - np.sin(normV) / normV) / normV**2
        tangentTensor = np.zeros(shape=[3,3], dtype=np.float64)
        eps = permutationOp()
        for i in range(3):
            for j in range(3):
                tangentTensor[i,j] = Identity[i,j] + term1 * (eps[i,:,j] @ v[:]) + term2 * (v[:] @ eps[:,i,:] @ eps[:,j,:] @ v[:])
        return tangentTensor

def interpolateRotations(rotationArray, lagrangeInterpolant):
    '''
    This function interpolates the given rotation array to Gauss points.
    It uses a reference rotation matrix to interpolate locally within an element and then uses the transpose of reference rotation matrix to get global interpolated rotation vectors.
    See paper by Crisfield and Jelenic, 1998 on 'Objectivity of strain measures in geometrically exact 3-D beam theory and it FE implementation.
    Also Algorithm 1 in paper 'Interpolation of rotational variables in nonlinear dynamics of 3D beams' by Jelenic, Crisfield 1998
    Argument:
        rotationArray: np.ndarray of shape NNPEL x 3. 
        lagrangeInterpolant: np.array of shape NNPEL. It is Lagrange interpolation function computed for the Gauss point where rotationArray needs to be computed.
    Return:
        interpRotVec: np.array of shape 3. The interpolated rotation vector at the Gauss point.
    '''

    NNPEL, _ = rotationArray.shape
    referenceRotationMatrix = np.zeros(shape=[3, 3], dtype=np.float64)
    localRotationVectors = np.zeros(shape=[NNPEL,3], dtype=np.float64)
    dummy = np.zeros(shape=[3,3], dtype=np.float64)
    
    for nnpel in range(NNPEL):
        rotMat = Rotation.from_rotvec(rotationArray[nnpel,:]).as_matrix()
        if nnpel == 0:
            referenceRotationMatrix[:,:] = rotMat
        dummy[:,:] = referenceRotationMatrix.T @ rotMat
        localRotationVectors[nnpel,:] = Rotation.from_matrix(dummy).as_rotvec()

    interpolatedRotationVec = lagrangeInterpolant @ localRotationVectors
    interpRotMat = referenceRotationMatrix @ Rotation.from_rotvec(interpolatedRotationVec).as_matrix()
    interpRotVec = Rotation.from_matrix(interpRotMat).as_rotvec()

    return interpRotVec

def plotzzz(x,y):
    'Quick plotting function'
    fig, ax = plt.subplots()
    ax.plot(y, x)
    plt.show()

    return None

def transformDimension(array:np.ndarray)->np.ndarray:
    '''
    This function transforms a given global vector with shape totalNodes * DOFPN. into a matrix of shape totalNodes x DOFPN.
    Argument:
        array: np.ndarray of shape (totalNodes * DOFPN,). The global array to be transformed.
    Return:
        matrix: np.ndarray of shape (totalNodes, DOFPN). The transformed matrix.
    '''
    DOFPN = 6
    rows = array.shape
    nodes = int(rows[0] / DOFPN)
    matrix = np.zeros(shape = [nodes, DOFPN], dtype=np.float64)

    for i in range(DOFPN):
        matrix[:,i] = array[i::DOFPN]
        
    return matrix

