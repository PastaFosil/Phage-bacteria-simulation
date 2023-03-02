import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, njit

#@jit
def wca(r_ij):
    global sigmav
    r_ij = r_ij.astype(np.float64)

    mask = r_ij!=0
    r_ij[mask] = (48/r_ij[mask])*((sigmav/r_ij[mask])**12-.5*(sigmav/r_ij[mask])**6)

    return r_ij

def electrostatic(r_ij):
    global eps, qv, qb
    mask = r_ij!=0
    r_ij[mask] = -eps*qv*qb/r_ij[mask]

    return r_ij

#@jit
def totalPairForce(pos, dist, ev):
    global sigmav, L
    #tree = cKDTree(posv,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    #dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
    #dist = dist.toarray()
    sh = pos.shape
    XYdist = np.ones((sh[0],sh[0],2))*pos #Posiciones de todas las particulas (un conjunto para cada particula)
    ref = np.ones((sh[0],sh[0],2)) #Posiciones repetidas (cada conjunto ayudara a calcular las componentes de la distancia de una particula dada a todas las demas)
    #for i in range(pos.shape[0]): #Dandole valores a los grupos de ref
    #    ref[i] *= pos[i]
    ref = duplicatePosition(ref, pos)

    XYdist[dist==0] = [0,0] #Ignorar las que estan fuera de la distancia de corte
    ref[dist==0] = [0,0] 

    mImgDist = ref-XYdist #Distancia entre pares (dirigido hacia particula de referencia)
    mImgDist = mImgDist - np.round_(mImgDist/L)*L #Distancia de minima imagen (vectores directores entre pares)
    unit = unitVector(mImgDist)
    dotFactor = dot(unit, np.array([ev]*sh[0]))
    dotFactor = checkOrient(dotFactor)

    forceWCA = wca(dist[:Nv, :Nv]) #Fuerza debida a WCA (sobre los virus)
    
    #forceElectro = np.zeros(mImgDist.shape)
    #forceElectro[-Nb:, :Nv] = np.ones((Nb,Nv))*mImgDist[-Nb:, :Nv]
    forceElectro = electrostatic(dist[-Nb:, :Nv])
    forceElectro[:Nv, -Nb:] = -forceElectro[-Nb:, :Nv].T
    #forceElectro[:Nv, -Nb:] = mImgDist[:Nv, -Nb:]
    
    mImgDist[:Nv, :Nv] *= np.transpose(np.array([forceWCA,]*2))
    mImgDist[-Nb:, :Nv] *= np.transpose(np.array([forceElectro,]*2))
    mImgDist[:Nv, -Nb:] *= np.transpose(np.array([-np.transpose(forceElectro),]*2))
    
    return np.sum(mImgDist, axis = 1)