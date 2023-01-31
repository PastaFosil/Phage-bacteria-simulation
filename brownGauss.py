import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit

def changePos(pos):
    tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
    dist = dist.toarray() #Matriz en formato np.array
    k = 0
    while np.any(dist[dist<sigma]) == True:
        reassign = np.where(np.logical_and(np.triu(dist)<1, np.triu(dist)>0))
        for i in reassign[0]:
            pos[i] = np.random.uniform(sigma/2, L-sigma/2, size = 2)
        tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
        dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
        dist = dist.toarray()
        k += 1
    return pos

#@jit
def wca(r_ij):
    global sigma
    r_ij = r_ij.astype(np.float64)

    mask = r_ij!=0
    r_ij[mask] = (48/r_ij[mask])*((sigma/r_ij[mask])**12-.5*(sigma/r_ij[mask])**6)
    #r_ij = np.where(r_ij>0, (48/r_ij)*((sigma/r_ij)**12-.5*(sigma/r_ij)**6), r_ij)
    #r_ij = np.piecewise(r_ij, [r_ij<=0.0, r_ij>0.0], [0.0, (48.0/r_ij)*((sigma/r_ij)**12-.5*(sigma/r_ij)**6)])
    return r_ij
#@jit
def totalPairForce(pos, dist):
    global sigma, L
    #tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    #dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
    #dist = dist.toarray()
    
    XYdist = np.ones((pos.shape[0],pos.shape[0],2))*pos
    ref = np.ones((pos.shape[0],pos.shape[0],2))
    for i in range(pos.shape[0]):
        ref[i] *= pos[i]

    XYdist[dist==0] = [0,0]
    ref[dist==0] = [0,0]

    mImgDist = ref-XYdist
    mImgDist = mImgDist - np.round_(mImgDist/L)*L
    #print(np.sum(dist))
    force = wca(dist)
    mImgDist *= np.transpose(np.array([force,]*2))
    
    return np.sum(mImgDist, axis = 1)

@jit
def MSD(pos_0, pos_t):
    msd = norm2(pos_0 - pos_t)
    return np.mean(msd)
@jit
def norm2(vec):
    return vec[:,0]**2+vec[:,1]**2

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
    
def animate(i): #Funcion ejecutada en cada cuadro (actualizacion de las posiciones y updateaciones
    global L,sigma, count, posT, pos, posNP

    #print(count)
    #print('pos', pos)
    update = np.random.normal(scale=1, size=(N,2))
    
    tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
    dist = tree.sparse_distance_matrix(tree, max_distance=rc,output_type='coo_matrix') #Matriz con los puntos mas cercanos
    dist = dist.toarray()
    LJ = totalPairForce(pos, dist)
    
    pos[:,0] += update[:,0]*(2*D*deltat)**(1/2)+(deltat*D/T)*LJ[:,0] #Actualizacion de la posicion en X de las particulas
    pos[:,1] += update[:,1]*(2*D*deltat)**(1/2)+(deltat*D/T)*LJ[:,1] #                             en Y

    posNP[:,0] += update[:,0]*(2*D*deltat)**(1/2)+(deltat*D/T)*LJ[:,0]
    posNP[:,1] += update[:,1]*(2*D*deltat)**(1/2)+(deltat*D/T)*LJ[:,1]
    
    print('t=', count*deltat)
    
    pos[pos>L] -= L #Circulacion de la posicion de las particulas que se salen de la caja
    pos[pos<0] += L

    if count%inter == 0:
        posT = np.insert(posT, int(count/inter), posNP, axis=0)
        
    count += 1

    #br.set_offsets(pos)
    #return br,

"------------------------------------------------------------------------"
# [thsigma t+1]=arg(suma exp(i [thsigma t]) )+[sigma][chi]
# ^orientacion t+1   ^angulo del vector promedio  ^orientacion del ruido 
"------------------------------------------------------------------------"
'================================================================================================'
'PARaMETROS'

sigma = 1 #Diametro de la particula
L = 30.0 #Tamano de la caja
phi = .05 #packing fraction
rho = 4*phi/(np.pi*sigma**2) #Densidad de particulas
#N = int(rho*L**2) #Numero de particulas
N = 1000
L = np.sqrt(N*np.pi/(4*phi))
print(" N",N)

deltat = 1e-4 #Intervalo de tiempo
inter = 5 #Iteraciones para el guardado de posiciones
iterations = 20000 #Numero de cuadros

T = 1.0 #Temperatura
D = .077 #Coeficiente de difusion
rc = sigma*2**(1/6) #Radio de corte

'================================================================================================'
'CONFIGURACIoN INICIAL'
'''
pos = np.random.uniform(sigma/2,L-sigma/2,size=(N,2)) #Arreglo de la posicion de las particulas
pos = changePos(pos) #Eliminacion de solapamiento
posNP = np.ones(shape=(N,2))*pos #Posicion sin condiciones periodicas de frontera
posT = np.array([pos]) #Posicion al tiempo t
'''
'''
fig, ax= plt.subplots(figsize=(6,6)) #Creacion de la figura y subtrama
ax.set_xlim([0,L])
ax.set_ylim([0,L])
br = ax.scatter(pos[:,0],pos[:,1])
'''
'================================================================================================'
'ANIMACIoN'
'''
count = 1
#anim = FuncAnimation(fig,animate,frames=iterations,interval=1, blit=True, repeat=False) #Generacion de la animacion
#3plt.show()

for i in range(iterations):
    animate(i)
'''
'================================================================================================'
'D*/D'

file = open('Deff_D', 'wb')
aux = np.array([])
for i in range(10):
    pos = np.random.uniform(sigma/2,L-sigma/2,size=(N,2)) #Arreglo de la posicion de las particulas
    pos = changePos(pos)
    posNP = np.ones(shape=(N,2))*pos #Posicion sin condiciones periodicas de frontera
    posT = np.array([pos]) #Posicion al tiempo t

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

D_eff = np.array([[phi, np.mean(aux/4)/D]])
np.save(file, D_eff)

'''
    for j in range(10):
        pos = np.random.uniform(sigma/2,L-sigma/2,size=(N,2)) #Arreglo de la posicion de las particulas
        pos = changePos(pos)
        posNP = np.ones(shape=(N,2))*pos #Posicion sin condiciones periodicas de frontera
        posT = np.array([pos]) #Posicion al tiempo t

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
'''
pMSD = np.array([])
temp = np.array([])
k = 0
for t in posT:
    pMSD = np.append(pMSD, MSD(posT[0], t))
    temp = np.append(temp, deltat*inter*(k+3))
    k += 1

nombre = 'phi_'+str(phi).replace('.', '_')
data = np.array([temp, pMSD])
file = open(nombre, 'wb')
np.save(file, data)
file.close

lin = np.polyfit(temp, pMSD, 1)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*D, lin[1]])

f, (a1,a2) = plt.subplots(1,2)

a1.scatter(temp,pMSD, color='k' )
a1.set_title('MSD')
a1.set_xlabel('Tiempo')
a1.set_ylabel('MSD')

lin = np.polyfit(temp, pMSD, 1)
#print(lin)
lin_model_sim = np.poly1d(lin)
lin_model_an = np.poly1d([4*D, lin[1]])

a1.plot(temp, lin_model_sim(temp),color='g')
a1.plot(temp, lin_model_an(temp),color = 'r')
'''
'''
#print(posT)
tree = cKDTree(pos,boxsize=[L,L]) #arbol de las particulas mas cercanas entre si
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