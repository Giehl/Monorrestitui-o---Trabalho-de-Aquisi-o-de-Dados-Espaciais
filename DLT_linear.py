# Declarar as bibliotecas------------------------------------------------------
import numpy as np
import math as mt 
import cv2 as cv
from sympy import *
from numpy.linalg import inv

# Distância focal--------------------------------------------------------------
c = 15.8893/1000

# Coordenadas do ponto principal ----------------------------------------------
xi = 0.1746/1000
yi = -0.0947/1000

# Parâmetros de distorção radial simétrica-------------------------------------
k1 = (-2.54848278e-04)/1000
k2 = (1.48901365e-06)/1000
k3 =  0.0

# Parâmetros de distorção descentrada------------------------------------------
p1 = (-8.92107195e-05)/1000
p2 = (-8.00528299e-05)/1000

# Dimensão do pixel no sensor em mm--------------------------------------------
tpx = 0.007/1000
tpy = 0.007/1000
tp= 0.007/1000

# Altura do voo aproximada-----------------------------------------------------
hvoo = 300

# Tamanho da imagem------------------------------------------------------------
U = 3344
T = 2224

# Nome do Ponto, X, Y, Z, linha, coluna----------------------------------------
pontos = [
    ('2', 488.402, 563.665, 8.267, 1047, 863),
    ('57', 444.191, 669.765, 21.42, 315, 129),
    ('84', 675.587, 374.946, 1.432, 1899, 2684),
    ('94', 525.366, 492.427, 7.269, 1476, 1337),
    ('108', 554.357, 662.66, 12.16, 88, 1074),
    ('123', 669.969, 577.359, 9.741, 456, 2206),
    ('127', 770.173, 537.567, 8.069, 503, 3056),
    ('129', 648.682, 432.527, 2.117, 1575, 2370),
    ('133', 585.134, 476.26, 3.493, 1432, 1816),
    ('136', 395.148, 530.693, 6.973, 1530, 278),
    ('138', 395.067, 588.307, 8.878, 1110, 84),
    ('145', 579.864, 510.927, 5.448, 1197, 1689)
    ]


#  Pontos de Controle na imagem-CL (02,57,84,127,136,145)----------------------
cl_pc = np.array([[863,1047],
       [129,315],
       [2684,1899],
       [3056,503],
       [278,1530],
       [1689,1197]])

# Pontos de verificação na imagem-CL (94,108,123,129,133,138)-----------------
cl_pv = np.array([[1337,1476],
         [1074,88],
         [2206,456],
         [2370,1575],
         [1816,1432],
         [84,1110]])

# Pontos de Controle no terreno - Nome do Ponto, XYZ (02,57,84,127,136,145)----
pa_pc = np.array([[2,488.402,563.665,8.267],
                  [57,444.191,669.765,21.42],
                  [84,675.587,374.946,1.432],
                  [127,770.173,537.567,8.069],
                  [136,395.148,530.693,6.973],
                  [145,579.864,510.927,5.448]])

# Pontos de Controle no terreno - Nome do Ponto, XYZ (94,108,123,129,133,138)--
pa_pv = np.array([[94,525.366,492.427,7.269],
                 [108,554.357,662.66,12.16],
                 [123,669.969,577.359,9.741],
                 [129,648.682,432.527,2.117],
                 [133,585.134,476.260,3.493],
                 [138,395.067,588.307,8.878]])

# Modelo funcional: DLT linear ------------------------------------------------
L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,x,y,X, Y, Z = symbols('L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 x y X Y Z')

xp = L1*X + L2*Y + L3*Z + L4 - x*L9*X - x*L10*Y - x*L11*Z
yp = L5*X + L6*Y + L7*Z + L8 - y*L9*X - y*L10*Y - y*L11*Z

dad=np.column_stack([cl_pc,pa_pc])

# Montar a matriz A -----------------------------------------------------------
def matrizA(dad,xp,yp):  
    a1 = diff(xp,L1)
    a2 = diff(xp,L2)
    a3 = diff(xp,L3)
    a4 = diff(xp,L4)
    a5 = diff(xp,L5)
    a6 = diff(xp,L6)
    a7 = diff(xp,L7)
    a8 = diff(xp,L8)
    a9 = diff(xp,L9)
    a10 = diff(xp,L10)
    a11 = diff(xp,L11)
    
    a12 = diff(yp,L1)
    a13 = diff(yp,L2)
    a14 = diff(yp,L3)
    a15 = diff(yp,L4)
    a16 = diff(yp,L5)
    a17 = diff(yp,L6)
    a18 = diff(yp,L7)
    a19 = diff(yp,L8)
    a20 = diff(yp,L9)
    a21 = diff(yp,L10)
    a22 = diff(yp,L11)
    
    coef_x = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]
    coef_y = [a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22]
    
    def IterA(coef,dad):
        val_A=[]
        for i in dad:
            for j in coef:
                num_a = j.evalf(subs={X: i[3], Y: i[4],\
                                      Z: i[5], x: i[0], y: i[1]})
                val_A=np.append(val_A,num_a)
        return val_A       
    
    val_Ax=IterA(coef_x,dad)
    val_Ay=IterA(coef_y,dad)
    
    A=np.matrix([val_Ax[0:11], val_Ay[0:11], val_Ax[11:22], val_Ay[11:22],\
                val_Ax[22:33], val_Ay[22:33],val_Ax[33:44],val_Ay[33:44],\
                val_Ax[44:55], val_Ay[44:55], val_Ax[55:66], val_Ay[55:66]])
    return A

x0= np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

val_x0=[]
for i in dad:
    for j in [xp,yp]:
        val =  j.evalf(subs={X: i[3], Y: i[4],\
                    Z: i[5], x: i[0], y: i[1],L1:x0[0][0],\
                    L2:x0[1][0],L3:x0[2][0],L4:x0[3][0],L5:x0[4][0],\
                    L6:x0[5][0],L7:x0[6][0],L8:x0[7][0],L9:x0[8][0],\
                    L10:x0[9][0],L11:x0[10][0]})
        val_x0=np.append(val_x0,val)     
L0=np.matrix(val_x0)
L0=np.transpose(L0)

# Vetor das observações Lb-----------------------------------------------------
pts_pc=[]
for i in cl_pc:
    pts_pc=np.append(pts_pc,(np.array([i[0],i[1]])))
Lb=np.matrix(pts_pc)
Lb=np.transpose(Lb)

# Matriz Peso------------------------------------------------------------------
I=np.identity(12, dtype=float)
#P=I*(1/((tp)**2))
#P=np.matrix(P)
P=I

# Matriz A---------------------------------------------------------------------
A=matrizA(dad,xp,yp).astype(np.float)       
At = np.transpose(A)

# Vetor dos parâmetros ajustados-----------------------------------------------
Xa=inv(At*P*A)*At*P*Lb

# Matriz dos resíduos----------------------------------------------------------
V=A*Xa-Lb

# Variância a posteriori-------------------------------------------------------
n=len(Lb)
u=len(Xa)
Vt=np.transpose(V)
sigma_pos=Vt*P*V/(n-u)

# MCV dos parâmetros-----------------------------------------------------------
mvc_p=float(sigma_pos)*inv(At*P*A)

# MVC dos valores ajustados----------------------------------------------------
mvc_vaj=A*inv(At*P*A)*At

xa=np.array(Xa)
d=np.column_stack([cl_pv,pa_pv])

# DLT na sua forma inversa ----------------------------------------------------
# Obtenção de X,Y (espaço objeto) dos pv por meio de  Xa (L1,L2. L3...L11), 
# x,y (objeto imagem) e Z dos pv ----------------------------------------------
pa_pv_cal=[]
for k in d:
    a11=xa[0][0]-k[0]*xa[8][0]
    a12=xa[1][0]-k[0]*xa[9][0]
    a21=xa[4][0]-k[1]*xa[8][0]
    a22=xa[5][0]-k[1]*xa[9][0]
    c11=-k[5]*(xa[2][0]-k[0]*xa[10][0])-xa[3][0]+k[0]
    c12=-k[5]*(xa[6][0]-k[1]*xa[10][0])-xa[7][0]+k[1]
    AA=np.matrix([[float(a11), float(a12)], [float(a21), float(a22)]])
    C=np.matrix([[float(c11)], [float(c12)]])
    pa_pvcal=np.dot(inv(AA),C)
    pa_pv_cal=np.append(pa_pv_cal,pa_pvcal)

pa_pv_vetor=[]
for i in pa_pv:
    X=i[1]
    Y=i[2]
    pa_pv_vetor=np.append(pa_pv_vetor,np.array([X,Y]))
    
# Diferença entre pv calculados e observados no espaço objeto------------------
pa_pv_dif=pa_pv_cal-pa_pv_vetor
print()
print()
print('------------------------------------')
print('Diferença entre pv calculados e observados no espaço objeto')
print()
print(pa_pv_dif)

# Análise da qualidade do ajustamento (qui-quadrado para alfa = 5%)------------
Qui_cal=Vt*P*V
Qui_tab_025=0.001
Qui_tab_975=5.024

print()
print()
print('------------------------------------')
print('Qui-quadrado para alfa = 5%')
print()
print(Qui_tab_025,'<',float(Qui_cal),'<', Qui_tab_975)

# Análise das precisões--------------------------------------------------------
#As raízes quadradas dos elementos da diagonal principal da matriz MVC das 
#observações ajustadas  irão fornecer a precisão das observações ajustadas ----
diag_mvc_p =np.array([row[i] for i,row in enumerate(np.array(mvc_p))])
presi=(diag_mvc_p)**(1/2)
print()
print()
print('------------------------------------')
print('Precisão das observações ajustadas')
print()
print(presi)

print()
print()
print('------------------------------------')
print('Valor T calculado > Valor T tabelado (alfa = 5%)')
print()
valor_t_tab=6.314
for i in range(0,len(Xa),1):
    valor_t_cal=abs(Xa[i][0])/((diag_mvc_p[i])**(1/2))
    print(float(valor_t_cal),'>',valor_t_tab)