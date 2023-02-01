import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


"------------------------------------------------------------------------"
# [theta t+1]=arg(suma exp(i [theta t]) )+[eta][chi]
# ^orientacion t+1   ^ángulo del vector promedio  ^orientación del ruido 
"------------------------------------------------------------------------"


L = 32.0 #Tamaño de la caja
rho = 3.0 #Densidad de partículas
N = int(rho*L**2) #Número de partículas
print(" N",N)
 
r0 = 1.0 #Radio de influencia
deltat = 1.0 
factor =0.5
v0 = r0/deltat*factor #Velocidad de las partículas
iterations = 10000 #Número de cuadros
eta = 0.145 #Amplitud del ruido

part = 16
R = 10
cirAng = np.zeros(shape=part)
circle = np.zeros(shape=(part, 2))
for i in range(part):
    cirAng[i] += np.pi*(-1+2*i/part)
    circle[i,0] = L/2 + R*np.cos(cirAng[i])
    circle[i,1] = L/2 + R*np.sin(cirAng[i])

pos = np.random.uniform(0,L,size=(N,2)) #Arreglo de la posición de las partículas
orient = np.random.uniform(-np.pi, np.pi,size=N) #Arreglo de la orientación de las partículas

fig, ax= plt.subplots(figsize=(6,6)) #Creación de la figura y subtrama
 
qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient[0]), np.sin(orient), orient, clim=[-np.pi, np.pi]) #Campo 2D de flechas
 #              X           y       Orientacion X       Y             color          vmin    vmax


qvCir = ax.quiver(circle[:,0], circle[:,1], np.cos(cirAng+np.pi/2), np.sin(cirAng+np.pi/2), cirAng, clim=[-np.pi, np.pi])
qv.Rad = ax.quiver(circle[:,0], circle[:,1], np.cos(cirAng+np.pi), np.sin(cirAng+np.pi), cirAng, clim=[-np.pi, np.pi])
def animate(i): #Función ejecutada en cada cuadro (actualización de las posiciones y orientaciones
    print(i)
 
    global orient

    tree = cKDTree(pos,boxsize=[L,L]) #Árbol de las partículas más cercanas entre sí
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix') #Matriz con los puntos más cercanos
    
    #important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j) #Términos de exponenciales complejas para cada partícula
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape()) #Asignación de las exponenciales a las partículas cercanas entre sí
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1))) #Sumatoria de los términos exponenciales para cada partícula y simplificación del arreglo
     
     
    orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N) #Actualización de las posiciones y orientaciones de las partículas
 
 
    cos, sin= np.cos(orient), np.sin(orient) #Nuevas orientaciones de las partículas
    pos[:,0] += cos*v0 #Actualización de la posición en X de las partículas
    pos[:,1] += sin*v0 #                             en Y
 
    pos[pos>L] -= L #Circulación de la posición de las partículas que se salen de la caja
    pos[pos<0] += L
 
    qv.set_offsets(pos) #Actualización de la posición de las partículas en la gráfica
    qv.set_UVC(cos, sin,orient) #Actualización de la orientación de las flechas en la gráfica
    return qv,

 
anim = FuncAnimation(fig,animate,np.arange(1, 200),interval=1, blit=True) #Generación de la animación

#animVect = FuncAnimation(fig,animateAtrac,np.arange(1, 200),interval=1, blit=True) #Generación de la animación

plt.show()
