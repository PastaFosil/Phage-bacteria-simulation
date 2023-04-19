import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, njit

#@jit
def uniquePos(Nv, Nb, sigmav, sigmab, L):
    refv = 4.*sigmav*sigmav
    refb = 4.*sigmab*sigmab
    pos = np.random.uniform(sigmav/2,L-sigmav/2,size=(1,2))
    for i in range(1,Nv):
        avant = 0
        i = 0
        while avant==0.:
            avant = 1.
            xaux = np.random.uniform(sigmav/2.,L-sigmav/2.,size=(1,2))
            for j in range(i):
                d = pos[j]-xaux[0]
                if np.dot(d,d)<refv:
                    avant = 0
        pos = np.append(pos,xaux,axis=0)


    for i in range(1,Nb):
        avant = 0
        while avant==0.:
            avant = 1.
            xaux = np.random.uniform(sigmav/2.,L-sigmav/2.,size=(1,2))
            for j in range(Nv+i):
                d = pos[j]-xaux[0]
                if np.dot(d,d)<refb:
                    avant = 0
        pos = np.append(pos,xaux,axis=0)
    return pos

#@jit
def wca(r_ij, sigma):
    r_ij = r_ij.astype(np.float64)

    mask = r_ij!=0
    r_ij[mask] = (48/r_ij[mask])*((sigmav/r_ij[mask])**12-.5*(sigmav/r_ij[mask])**6)

    return r_ij
"""
def electrostatic(r_ij):
    global eps, qv, qb
    mask = r_ij!=0
    r_ij[mask] = -eps*qv*qb/(r_ij[mask]**2)

    return r_ij
"""
#@jit
def totalPairForce(pos, dist, ev):
    global sigmav, L, sigmab, sigmav, rcFactor
    sigmabv = sigmav/2 + sigmab/2

    dist[:Nv,:Nv][dist[:Nv,:Nv]>rcFactor*sigmav] = 0
    dist[-Nb:,:Nv][dist[-Nb:,:Nv]>rcFactor*sigmabv] = 0
    dist[:Nv,-Nb:] = dist[-Nb:,:Nv].T
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

    mImgDist = ref-XYdist #Vector entre pares (dirigido hacia particula de referencia)
    mImgDist = mImgDist - np.round_(mImgDist/L)*L #Distancia de minima imagen (vectores directores entre pares)
    unit = unitVector(mImgDist[-Nb:,:Nv]) #Vector unitario entre pares (saliendo de las bacterias)
    dotFactor = dot(unit, np.array([ev]*Nb)) #Producto de vectores unitarios bacteria-virus con la orientación de los virus
    dotFactor = checkOrient(dotFactor) #Anulación de los productos con dot(ev,eb) >= 0

    forceWCAvv = wca(dist[:Nv, :Nv], sigmav) #Fuerza debida a WCA (sobre los virus)
    #print(np.max(forceWCA))
    #forceElectro = np.zeros(mImgDist.shape)
    #forceElectro[-Nb:, :Nv] = np.ones((Nb,Nv))*mImgDist[-Nb:, :Nv]
    forceWCAbv = wca(dist[-Nb:, :Nv], sigmabv)
    forceWCAbb = wca(dist[-Nb:, -Nb:], sigmab)
    #forceElectro[:Nv, -Nb:] = -forceElectro[-Nb:, :Nv].T
    #forceElectro[:Nv, -Nb:] = mImgDist[:Nv, -Nb:]
    
    mImgDist[:Nv, :Nv] *= np.transpose(np.array([forceWCAvv,]*2))
    mImgDist[-Nb:, :Nv] *= np.transpose(np.array([forceWCAbv,]*2),axes=[1,2,0])*dotFactor
    mImgDist[:Nv, -Nb:] = -np.transpose(mImgDist[-Nb:, :Nv],axes=[1,0,2])
    mImgDist[-Nb:, -Nb:] *= np.transpose(np.array([forceWCAbb,]*2))
    
    return np.sum(mImgDist, axis = 1)
@njit
def checkOrient(u):
    sh = np.shape(u)
    for g in range(sh[0]):
        for v in range(sh[1]):
            if u[g,v,0]<0.:
                u[g,v,0] = 0.
    return u

@njit
def dot(v1, v2):
    if v1.ndim==2:
        d = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1] #producto escalar de arreglo de vectores de Nx2
    else:
        sh = np.shape(v1) #arreglo de vectores NxNx2
        d = np.array([0.])
        for i in range(np.shape(v1)[0]):
            d = np.append(d, dot(v1[i],v2[i])) #arreglo unidimensional de los productos
        return d[1:].reshape((sh[0],sh[1],1)) #devolucion de arreglo NxNx1
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
    if v.ndim==2:
        norm = np.sqrt(dot(v,v))
        l = len(norm)
        norm = norm.reshape((l,1))
        #norm = reshapeRow(norm)
        for i in range(l):
            if norm[i]!=0:
                v[i] = v[i]/norm[i]
    else:
        sh = np.shape(v)
        d = np.array([[0.,0.]])
        for i in range(sh[0]):
            d = np.append(d, unitVector(v[i]), axis=0) #arreglo unidimensional de los productos
        return d[1:].reshape(sh) #devolucion de arreglo NxNx1

        v = v.reshape((sh[0]*sh[1],2))
    #norm = np.sqrt(dot(v)[np.newaxis].T)
    return v
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
        k[i] = np.i_nonzero(d)
    #print('k2: ', k)
    return k*(L/dist.shape[0])**2

#def selfPropulsion()    
def animate(i): #Funcion ejecutada en cada cuadro (actualizacion de las posiciones y orientaciones
    global L, sigmav, sigmab, Fa, rcFactor, orient, posT

    print('t=', i*deltat)
    #print(i)
    #print('posv', posv)
    stocastic = np.random.normal(scale=1, size=(Nv+Nb,2))
    stocasticR = np.random.uniform(-np.pi, np.pi, size=Nv+Nb)

    orient[:Nv] += stocasticR[:Nv]*stocRV
    orient[-Nb:] += stocasticR[-Nb:]*stocRB
    orient = np.clip(orient, -np.pi, np.pi)
    ev = np.zeros((Nv, 2))
    ev[:,0] = np.cos(orient[:Nv])
    ev[:,1] = np.sin(orient[:Nv])

    tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    dist = tree.sparse_distance_matrix(tree, max_distance=rcFactor*np.max([sigmav,sigmab]),output_type='coo_matrix') #Matriz con los puntos mas cercanos
    dist = dist.toarray()
    
    #LJ = totalPairForce(pos, dist, ev)
    
    #pos[:Nv,0] += stocastic[:Nv,0]*(2*D*deltat)**(1/2)+stocastic[-Nb:,0]*(2*Db*deltat)**(1/2)+(deltat*D/T)*LJ[:,0] #Actualizacion de la posicion en X de las particulas
    #pos[:Nv,1] += stocastic[:Nv,1]*(2*D*deltat)**(1/2)+stocastic[-Nb:,1]*(2*Db*deltat)**(1/2)+(deltat*D/T)*LJ[:,1] #                             en Y

    pos[:Nv,0] += stocastic[:Nv,0]*stocV# + frcV*LJ[:Nv,0] #Actualizacion de la posicion en X de las particulas
    pos[:Nv,1] += stocastic[:Nv,1]*stocV# + frcV*LJ[:Nv,1] #                             en Y
    posNP[:Nv,0] += stocastic[:Nv,0]*stocV# + frcV*LJ[:Nv,0]
    posNP[:Nv,1] += stocastic[:Nv,1]*stocV# + frcV*LJ[:Nv,1]

    #print(orient[-Nb:])
    pos[-Nb:,0] += stocastic[-Nb:,0]*stocB + frcB*Fa*np.cos(orient[-Nb:])# + frcB*LJ[-Nb:,0]
    pos[-Nb:,1] += stocastic[-Nb:,1]*stocB + frcB*Fa*np.sin(orient[-Nb:])# + frcB*LJ[-Nb:,1]
    posNP[-Nb:,0] += stocastic[-Nb:,0]*stocB + frcB*Fa*np.cos(orient[-Nb:])# + frcB*LJ[-Nb:,0]
    posNP[-Nb:,1] += stocastic[-Nb:,1]*stocB + frcB*Fa*np.sin(orient[-Nb:])# + frcB*LJ[-Nb:,1]
    #posNP[:,0] += stocastic[:,0]*(2*D*deltat)**(1/2)
    
    #posNP[:,1] += stocastic[:,1]*(2*D*deltat)**(1/2)
    
    pos[pos>L] -= L #Circulacion de la posicion de las particulas que se salen de la caja
    pos[pos<0] += L

    if i%inter == 0:
        posT = np.insert(posT, int(i/inter), posNP, axis=0)
        
    i += 1

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
#L = 30.0 #Tamano de la caja

sigmav = 1 #Diametro de los virus
phiv = .05 #packing fraction virus
rhov = 4*phiv/(np.pi*sigmav**2) #Densidad de particulas
#Nv = int(rhov*L**2) #Numero de particulas
Nv = 0
qv = 1 #Carga electrica
D = .077 #Coeficiente de difusion virus
Drv = 3*D/sigmav**2 #Coeficiente de difusion orientacional virus
stocV = (2*D*deltat)**(1/2) #Magnitud del termino estocastico (virus)
stocRV = (2*Drv*deltat)**(1/2) #Magnitud del termino estocastico rotacional (virus)
frcV = deltat*D/T #Factor de terminos de interacciones isotropa y anisotropa

sigmab = 10 #Diametro de las bacterias
qb = 1 #Carga electrica
phib = .05 #packing fraction bacterias
Nb = 1000
Db = .077 #Coeficiente de difusion bacteria
Drb = 3*D/sigmab**2 #Coeficiente de difusion orientacional virus
stocB = (2*Db*deltat)**(1/2) #Magnitud del termino estocastico (bacteria)
stocRB = (2*Drb*deltat)**(1/2) #Magnitud del termino estocastico rotacional (bacteria)
frcB = deltat*Db/T #Factor de terminos de autopropulsion e interaccion anisotropa
Fa = 10
#L = np.sqrt(1.5*Nv*np.pi/(4*phiv))
L = 1000
rcFactor = 2**(1/6) #Radio de corte

print(" Nv %s\n Nb %s" % (Nv, Nb))

'================================================================================================'
'CONFIGURACIoN INICIAL'

#pos = np.array([[1.1,1.4],[2.1,1.],[2.,1.9]])
"""
posv = np.random.uniform(sigmav/2,L-sigmav/2,size=(Nv,2)) #Arreglo de la posicion de las particulas
posv = changePos(posv) #Eliminacion de solapamiento

posb = np.random.uniform(sigmab/2,L-sigmab/2,size=(Nb,2)) #Arreglo de la posicion de las particulas
posb = changePos(posb) #Eliminacion de solapamiento

pos = np.append(posv, posb, axis=0) #posicion de las particulas (virus y bacterias)
"""

pos = uniquePos(Nv, Nb, sigmav, sigmab, L)
posNP = np.ones(shape=(Nv+Nb, 2))*pos #Posicion sin condiciones periodicas de frontera
posT = np.array([pos]) #Posicion al tiempo t
orient = np.random.uniform(-np.pi, np.pi, size=Nv+Nb) #Orientacion de las particulas

fig, ax= plt.subplots(figsize=(6,6)) #Creacion de la figura y subtrama
ax.set_xlim([0,L])
ax.set_ylim([0,L])

brv = ax.scatter(pos[:Nv,0],pos[:Nv,1], s=sigmav*10)
brb = ax.scatter(pos[-Nb:,0],pos[-Nb:,1], s=sigmab*10)

'================================================================================================'
'ANIMACIoN'

i = 1

anim = FuncAnimation(fig,animate,frames=iterations,interval=1, blit=True, repeat=False) #Generacion de la animacion
plt.show()
#ACTUALIZACION CADA 1000 PASOS
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

    i = 1
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

        i = 1
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
    #pMSD = np.append(pMSD, MSD(posT[0,:Nv,:], t[:Nv,:]))
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

f, (a1,a2) = plt.subplots(1,2)
"""
lin = np.polyfit(temp, pMSD, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*D, lin[1]])

a1.scatter(temp,pMSD, color='k' )
a1.set_title('MSD')
a1.set_xlabel('Tiempo')
a1.set_ylabel('MSD')

a1.plot(temp, lin_model_sim(temp),color='g')
a1.plot(temp, lin_model_an(temp),color = 'r')
"""


lin = np.polyfit(temp, pMSDb, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*Db, lin[1]])

a2.scatter(temp,pMSDb, color='k' )
a2.set_title('MSD')
a2.set_xlabel('Tiempo')
a2.set_ylabel('MSD')

a2.plot(temp, lin_model_sim(temp),color='g')
a2.plot(temp, lin_model_an(temp),color = 'r')
"""
lin = np.polyfit(temp, pMSDb, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*Db, lin[1]])

a3.scatter(temp,pMSDb, color='k' )
a3.set_title('MSD')
a3.set_xlabel('Tiempo')
a3.set_ylabel('MSD')

a3.plot(temp, lin_model_sim(temp),color='g')
a3.plot(temp, lin_model_an(temp),color = 'r')
"""
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