import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, njit

def changePos(posv):
    tree = cKDTree(posv,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
    dist = dist.toarray() #Matriz en formato np.array
    k = 0
    while np.any(dist[dist<sigmav]) == True:
        reassign = np.where(np.logical_and(np.triu(dist)<1, np.triu(dist)>0))
        for i in reassign[0]:
            posv[i] = np.random.uniform(sigmav/2, L-sigmav/2, size = 2)
        tree = cKDTree(posv,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
        dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
        dist = dist.toarray()
        k += 1
    return posv

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
@njit
def checkOrient(u):
    for i in range(len(u)):
        if u[i]>=0:
            u[i] = 0
    return u
@njit
def dot(v1, v2):
    return v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]
@njit
def dotVectorGroups(v1,v2):
    sh = np.shape(v1)
    v1 = v1.reshape((sh[0]*sh[1],2))
    v2 = v2.reshape((sh[0]*sh[1],2))
    d = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]
    return d.reshape((sh[0],sh[1],1))

@njit
def dd(v1, v2):
    if v1.ndim==2:
        d = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]
    else:
        sh = np.shape(v1)
        v1 = v1.reshape((sh[0]*sh[1],2))
        v2 = v2.reshape((sh[0]*sh[1],2))
        d = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]
        #d = reshapeRow(d, sh)
    return d


@njit
def duplicatePosition(ref, pos):
    for i in range(pos.shape[0]): #Dandole valores a los grupos de ref
        ref[i] *= pos[i]
    return ref

@njit
def MSD(pos_0, pos_t):
    v = pos_0 - pos_t
    msd = dot(v,v)
    return np.mean(msd)

@njit
def unitVector(v):
    sh = np.shape(v)
    v = v.reshape((sh[0]*sh[1],2))
    uv = np.zeros((sh[0]*sh[1],2))
    #norm = np.sqrt(dot(v)[np.newaxis].T)
    norm = np.sqrt(dot(v,v))
    l = len(norm)
    norm = norm.reshape((l,1))
    #norm = reshapeRow(norm)
    for i in range(l):
        if norm[i]!=0:
            uv[i] = v[i]/norm[i]

    return uv.reshape((sh[0],sh[1],2))
@njit
def reshapeRow(v, sh):
    new = np.zeros((sh[0],sh[1],1))
    for g in range(sh[0]):
        for vec in range(sh[1]):
            new[g,vec] = v[g+vec,0]
    return new
#@njit
#def re
def g(dist, r, dr):
    global L

    k = np.arange(0, L/2, dr)
    #print('k1: ', k)
    
    for i in range(k.shape[0]-1):
        d = np.ones(dist.shape)*dist
        d[d<k[i]] = 0
        d[d>k[i+1]] = 0
        k[i] = np.count_nonzero(d)
    #print('k2: ', k)
    return k*(L/dist.shape[0])**2

#def selfPropulsion()    
def animate(i): #Funcion ejecutada en cada cuadro (actualizacion de las posiciones y orientaciones
    global L,sigmav, count, posT, posv, posNP, orient

    #print(count)
    #print('posv', posv)
    stocastic = np.random.normal(scale=1, size=(Nv+Nb,2))
    stocasticR = np.random.uniform(-np.pi, np.pi, size=Nv+Nb)
    
    orient[:Nv] += stocasticR[:Nv]*stocRV
    orient[-Nb:] += stocasticR[-Nb:]*stocRB
    orient = np.clip(orient, -np.pi, np.pi)
    ev = np.zeros((Nv, 2))
    ev[:,0] = np.cos(orient(:Nv))
    ev[:,1] = np.sin(orient(:Nv))

    tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
    dist = dist.toarray()
    #LJ = totalPairForce(pos, dist, ev)
    
    #pos[:Nv,0] += stocastic[:Nv,0]*(2*D*deltat)**(1/2)+stocastic[-Nb:,0]*(2*Db*deltat)**(1/2)+(deltat*D/T)*LJ[:,0] #Actualizacion de la posicion en X de las particulas
    #pos[:Nv,1] += stocastic[:Nv,1]*(2*D*deltat)**(1/2)+stocastic[-Nb:,1]*(2*Db*deltat)**(1/2)+(deltat*D/T)*LJ[:,1] #                             en Y

    pos[:Nv,0] += stocastic[:Nv,0]*stocV# + frcV*LJ[:Nv,0] #Actualizacion de la posicion en X de las particulas
    pos[:Nv,1] += stocastic[:Nv,1]*stocV# + frcV*LJ[:Nv,1] #                             en Y
    posNP[:Nv,0] += stocastic[:Nv,0]*stocV# + frcV*LJ[:,0]
    posNP[:Nv,1] += stocastic[:Nv,1]*stocV# + frcV*LJ[:,1]

    print(orient[-Nb:])
    pos[-Nb:,0] += stocastic[-Nb:,0]*stocB + frcB*np.cos(orient[-Nb:])
    pos[-Nb:,1] += stocastic[-Nb:,1]*stocB + frcB*np.sin(orient[-Nb:])
    posNP[-Nb:,0] += stocastic[-Nb:,0]*stocB + frcB*np.cos(orient[-Nb:])
    posNP[-Nb:,1] += stocastic[-Nb:,1]*stocB + frcB*np.sin(orient[-Nb:])
    #posNP[:,0] += stocastic[:,0]*(2*D*deltat)**(1/2)
    
    #posNP[:,1] += stocastic[:,1]*(2*D*deltat)**(1/2)
    
    print('t=', count*deltat)
    
    pos[pos>L] -= L #Circulacion de la posicion de las particulas que se salen de la caja
    pos[pos<0] += L

    if count%inter == 0:
        posT = np.insert(posT, int(count/inter), posNP, axis=0)
        
    count += 1

    brv.set_offsets(pos[:Nv])
    brb.set_offsets(pos[-Nb:])
    return brb, brv,

"------------------------------------------------------------------------"
# [thsigmav t+1]=arg(suma exp(i [thsigmav t]) )+[sigmav][chi]
# ^orientacion t+1   ^angulo del vector promedio  ^orientacion del ruido 
"------------------------------------------------------------------------"
'================================================================================================'
'PARaMETROS'

deltat = 1e-4 #Intervalo de tiempo
inter = 5 #Iteraciones para el guardado de posiciones
iterations = 20000 #Numero de cuadros

T = 1.0 #Temperatura
eps = 1 #Constante electrica

sigmav = 1 #Diametro de los virus
phiv = .05 #packing fraction virus
#Nv = int(rhov*L**2) #Numero de particulas
Nv = 500
qv = 1 #Carga electrica
rhov = 4*phiv/(np.pi*sigmav**2) #Densidad de particulas
D = .077 #Coeficiente de difusion virus
Drv = 3*D/sigmav**2 #Coeficiente de difusion orientacional virus
stocV = (2*D*deltat)**(1/2) #Magnitud del termino estocastico (virus)
stocRV = (2*Drv*deltat)**(1/2) #Magnitud del termino estocastico rotacional (virus)
frcV = deltat*D/T #Factor de terminos de interacciones isotropa y anisotropa

sigmab = 10 #Diametro de las bacterias
qb = 1 #Carga electrica
phib = .05 #packing fraction bacterias
Nb = 50
Db = .077 #Coeficiente de difusion bacteria
Drb = 3*D/sigmab**2 #Coeficiente de difusion orientacional virus
stocB = (2*Db*deltat)**(1/2) #Magnitud del termino estocastico (bacteria)
stocRB = (2*Drb*deltat)**(1/2) #Magnitud del termino estocastico rotacional (bacteria)
frcB = deltat*Db/T #Factor de terminos de autopropulsion e interaccion anisotropa

rc = sigmav*2**(1/6) #Radio de corte
#L = 30.0 #Tamano de la caja
L = np.sqrt(Nv*np.pi/(4*phiv))
print(" Nv %s\n Nb %s" % (Nv, Nb))

'================================================================================================'
'CONFIGURACIoN INICIAL'

posv = np.random.uniform(sigmav/2,L-sigmav/2,size=(Nv,2)) #Arreglo de la posicion de las particulas
posv = changePos(posv) #Eliminacion de solapamiento

posb = np.random.uniform(sigmab/2,L-sigmab/2,size=(Nb,2)) #Arreglo de la posicion de las particulas
posb = changePos(posb) #Eliminacion de solapamiento

pos = np.append(posv, posb, axis=0) #posicion de las particulas (virus y bacterias)
posNP = np.ones(shape=(Nv+Nb, 2))*pos #Posicion sin condiciones periodicas de frontera
posT = np.array([pos]) #Posicion al tiempo t
orient = np.random.uniform(-np.pi, np.pi, size=Nv+Nb) #Orientacion de las particulas

fig, ax= plt.subplots(figsize=(6,6)) #Creacion de la figura y subtrama
ax.set_xlim([0,L])
ax.set_ylim([0,L])

brv = ax.scatter(posv[:Nv,0],posv[:Nv,1], s=sigmav*10)
brb = ax.scatter(posv[-Nb:,0],posv[-Nb:,1], s=sigmab*10)

'================================================================================================'
'ANIMACIoN'

count = 1

anim = FuncAnimation(fig,animate,frames=iterations,interval=1, blit=True, repeat=False) #Generacion de la animacion
plt.show()
"""
for i in range(iterations):
    animate(i)
"""
'================================================================================================'
'D*/D'
"""
file = open('Deff_D', 'wb')
aux = np.array([])
for i in range(10):
    posv = np.random.uniform(sigmav/2,L-sigmav/2,size=(Nv,2)) #Arreglo de la posicion de las particulas
    posv = changePos(posv)
    posNPv = np.ones(shape=(Nv,2))*posv #Posicion sin condiciones periodicas de frontera
    posT = np.array([posv]) #Posicion al tiempo t

    count = 1
    for i in range(iterations):
        animate(i)
    
    pMSD = np.array([])
    temp = np.array([])
    k = 0
    for t in posT:
        pMSD = np.append(pMSD, MSD(posT[0], t))
        temp = np.append(temp, deltat*inter*(k+3))
        k += 1

    lin = np.polyfit(temp, pMSD, 1)
    aux = np.append(aux, lin[0])

D_eff = np.array([[phiv, np.mean(aux/4)/D]])
np.save(file, D_eff)
"""
'''
    for j in range(10):
        posv = np.random.uniform(sigmav/2,L-sigmav/2,size=(Nv,2)) #Arreglo de la posicion de las particulas
        posv = changePos(posv)
        posNPv = np.ones(shape=(Nv,2))*posv #Posicion sin condiciones periodicas de frontera
        posT = np.array([posv]) #Posicion al tiempo t

        count = 1
        #anim = FuncAnimation(fig,animate,frames=iterations,interval=1, blit=True, repeat=False) #Generacion de la animacion
        for i in range(iterations):
            animate(i)

        pMSD = np.array([])
        temp = np.array([])
        k = 0
        
        auxD[j] = lin[0]/4
        
    disp = np.append(disp, np.mean(auxD))
disp /= D
x = np.linspace(.05, .35,50)

f, a = plt.subplots(1,1)

lin = np.polyfit(x, disp, 1)
lin_model_sim = np.poly1d(lin)

a.scatter(x,disp, color='k')
a.plot(x, lin_model_sim(x),color='g')
plt.show()
'''
'================================================================================================'
'ANaLISIS DE DATOS'

pMSD = np.array([])
pMSDb = np.array([])
pMSDtotal = np.array([])
temp = np.array([])
k = 0
for t in posT:
    pMSD = np.append(pMSD, MSD(posT[0,:Nv,:], t[:Nv,:]))
    pMSDb = np.append(pMSDb, MSD(posT[0,-Nb:,:], t[-Nb:,:]))
    pMSDtotal = np.append(pMSDtotal, MSD(posT[0], t))
    temp = np.append(temp, deltat*inter*(k+3))
    k += 1
"""
nombre = 'phiv_'+str(phiv).replace('.', '_')
data = np.array([temp, pMSD])
file = open(nombre, 'wb')
np.save(file, data)
file.close
"""

f, (a1,a2,a3) = plt.subplots(1,3)

lin = np.polyfit(temp, pMSD, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*D, lin[1]])

a1.scatter(temp,pMSD, color='k' )
a1.set_title('MSD')
a1.set_xlabel('Tiempo')
a1.set_ylabel('MSD')

a1.plot(temp, lin_model_sim(temp),color='g')
a1.plot(temp, lin_model_an(temp),color = 'r')



lin = np.polyfit(temp, pMSDb, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*Db, lin[1]])

a2.scatter(temp,pMSDb, color='k' )
a2.set_title('MSD')
a2.set_xlabel('Tiempo')
a2.set_ylabel('MSD')

a2.plot(temp, lin_model_sim(temp),color='g')
a2.plot(temp, lin_model_an(temp),color = 'r')

lin = np.polyfit(temp, pMSDb, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*Db, lin[1]])

a3.scatter(temp,pMSDb, color='k' )
a3.set_title('MSD')
a3.set_xlabel('Tiempo')
a3.set_ylabel('MSD')

a3.plot(temp, lin_model_sim(temp),color='g')
a3.plot(temp, lin_model_an(temp),color = 'r')

plt.show()
'''
#print(posT)
tree = cKDTree(posv,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
dist = tree.sparse_distance_matrix(tree, max_distance=L/2,output_type='coo_matrix') #Matriz con los puntos mas cercanos
dist = dist.toarray()
dr = .1
pairDist = g(dist, L/2, dr)
x = np.arange(0, L/2, dr)

a2.plot(x, pairDist)
a2.set_title('g(r)')
a2.set_xlabel('Distancia')
a2.set_ylabel('g(r)')
'''
'''
a2.axhline(D)
a2.scatter(temp,np.append(pMSD[0],pMSD[1:]/(4*temp[1:])), color='k' )
a2.set_title('D*')
a2.set_xlabel('Tiempo')
a2.set_ylabel('D*/4t')

plt.show()
'''