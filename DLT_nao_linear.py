
# Declarar as bibliotecas------------------------------------------------------
import numpy as np
import math as mt 
import cv2 as cv
from sympy import *
from numpy.linalg import inv, pinv
np.set_printoptions(suppress=True)

#DistÃ¢ncia focal---------------------------------------------------------------
c = 15.8893

#Coordenadas do ponto principal------------------------------------------------
xi = 0.1746
yi = -0.0947

# ParÃ¢metros de distorÃ§Ã£o radial simÃ©trica-------------------------------------
k1 = (-2.54848278e-04)
k2 = (1.48901365e-06)
k3 =  0.0

# ParÃ¢metros de distorÃ§Ã£o descentrada------------------------------------------
p1 = (-8.92107195e-05)
p2 = (-8.00528299e-05)

# DimensÃ£o do pixel no sensor em mm--------------------------------------------
tpx = 0.007
tpy = 0.007
tp= 0.007

# Altura do voo aproximada-----------------------------------------------------
hvoo = 300

# Tamanho da imagem------------------------------------------------------------
U = 3344
T = 2224

'''Pontos de controle: 57,84,108,127,136,145'''
#  Pontos de Controle na imagem-CL (57,84,108,127,136,145)----------------------
cl_pc = np.array([[129,316],
       [2686,1900],
       [1075,89],
       [3056,504],
       [279,1531],
       [1692,1199]])

# Pontos de controle no terreno (Nome do Ponto, XYZ (57,84,108,127,136,145)----
pa_pc = np.array([[57,444.191,669.765,21.42],
                  [84,675.587,374.946,1.432],
                  [108,554.357,662.66,12.16],
                  [127,770.173,537.567,8.069],
                  [136,395.148,530.693,6.973],
                  [145,579.864,510.927,5.448]])


'''Pontos de verificaÃ§Ã£o: 2,94,123,129,133,138'''
# Pontos de verificaÃ§Ã£o na imagem-CL (2,94,123,129,133,138)--------------------
cl_pv = np.array([[865,1048],
        [1338,1478],
         [2207,458],
         [2373,1577],
         [1816,1434],
         [85,1111]])


# Pontos de verificaÃ§Ã£o no terreno (Nome do Ponto, XYZ (02,94,123,129,133,138))
pa_pv=np.array([[2,488.402,563.665,8.267],
           [94,525.366,492.427,7.269],
           [123,669.969,577.359,9.741],
           [129,648.682,432.527,2.117],
           [133,585.134,476.26,3.493],
           [138,395.067,588.307,8.878]])


# média
Xm=[]
Ym=[]
Zm=[]
for i in pa_pc:
    Xm=np.append(Xm,i[1])
    Ym=np.append(Ym,i[2])
    Zm=np.append(Zm,i[3])
    Xm=np.mean(Xm)
    Ym=np.mean(Ym)
    Zm=np.mean(Zm)

def CL_to_xp(tpx,tpy,coluna,linha,U,T):
    #coordenadas sem correção (convertendo para o referencial com origem no centro da imagem)
    xl = tpx * (coluna - (U-1)/2)
    yl = -tpy * (linha - (T-1)/2)
    return xl,yl


# Pontos no referencial digital em mm (pontos de controle)---------------------
xy_ref_foto_pc=[]
for i in cl_pc:
    xy_Ref_fot=CL_to_xp(tpx,tpy,i[0],i[1],U,T)
    xy_ref_foto_pc=np.append(xy_ref_foto_pc,xy_Ref_fot)


# Pontos no referencial digital em mm (pontos de verificação)------------------
x_ref_foto_pv=[]
y_ref_foto_pv=[]
for i in cl_pv:
    xp1,yp1=CL_to_xp(tpx,tpy,i[0],i[1],U,T)
    x_ref_foto_pv=np.append(x_ref_foto_pv,np.array([xp1]))
    y_ref_foto_pv=np.append(y_ref_foto_pv,np.array([yp1]))
xy_ref_foto_pv=np.column_stack([x_ref_foto_pv,y_ref_foto_pv])

# Modelo funcional: DLT não linear --------------------------------------------

L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,X, Y, Z = symbols('L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 X Y Z')

xp=(L1*X+L2*Y+L3*Z+L4)/(L9*X+L10*Y+L11*Z+1)
yp =(L5*X+L6*Y+L7*Z+L8)/(L9*X+L10*Y+L11*Z+1)

# Montar a matriz A -----------------------------------------------------------
def matrizA(pa_pc,xp,yp,X0):  
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
    
    def IterA(coef,pa_pc,X0):
        val_A=[]
        for i in pa_pc:
            for j in coef:
                num_a=j.evalf(subs={X: i[1], Y: i[2],\
                    Z:i[3],L1:X0[0][0],\
                    L2:X0[1][0],L3:X0[2][0],L4:X0[3][0],L5:X0[4][0],\
                    L6:X0[5][0],L7:X0[6][0],L8:X0[7][0],L9:X0[8][0],\
                    L10:X0[9][0],L11:X0[10][0]})
                val_A=np.append(val_A,num_a)
        return val_A       
    
    val_Ax=IterA(coef_x,pa_pc,X0)
    val_Ay=IterA(coef_y,pa_pc,X0)
    
    A=np.matrix([val_Ax[0:11], val_Ay[0:11], val_Ax[11:22], val_Ay[11:22],\
                val_Ax[22:33], val_Ay[22:33],val_Ax[33:44],val_Ay[33:44],\
                val_Ax[44:55], val_Ay[44:55], val_Ax[55:66], val_Ay[55:66]])
    return A

# Estimação dos parâmetros iniciais X0, L0-------------------------------------
Ax=[]
Ay=[]
pa_cl_pc=np.column_stack([pa_pc,cl_pc])
for i in pa_cl_pc:
    a1x=i[1]
    a2x=i[2]
    a3x=i[3]
    a4x=1
    a5x=0
    a6x=0
    a7x=0
    a8x=0
    a9x=CL_to_xp(tpx,tpy,i[4],i[5],U,T)[0]*i[1]
    a10x=CL_to_xp(tpx,tpy,i[4],i[5],U,T)[0]*i[2]
    a11x=CL_to_xp(tpx,tpy,i[4],i[5],U,T)[0]*i[3]

    a1y=0
    a2y=0
    a3y=0
    a4y=0
    a5y=i[1]
    a6y=i[2]
    a7y=i[3]
    a8y=1
    a9y=CL_to_xp(tpx,tpy,i[4],i[5],U,T)[1]*i[1]
    a10y=CL_to_xp(tpx,tpy,i[4],i[5],U,T)[1]*i[2]
    a11y=CL_to_xp(tpx,tpy,i[4],i[5],U,T)[1]*i[3]
    
    Ax=np.append(Ax,np.column_stack([a1x,a2x,a3x,a4x,a5x,a6x,a7x,a8x,a9x,a10x,a11x]))
    Ay=np.append(Ay,np.column_stack([a1y,a2y,a3y,a4y,a5y,a6y,a7y,a8y,a9y,a10y,a11y]))
    
AA=np.matrix([Ax[0:11], Ay[0:11], Ax[11:22], Ay[11:22],\
               Ax[22:33], Ay[22:33], Ax[33:44],Ay[33:44],\
               Ax[44:55], Ay[44:55], Ax[55:66], Ay[55:66]])
    
B=np.matrix(xy_ref_foto_pc)
B=np.transpose(B)
X0 = pinv(AA)*B   
X0=np.array(X0)

val_L0=[]
for i in pa_pc:
    for j in [xp,yp]:
        val =  j.evalf(subs={X: i[1], Y: i[2],\
                    Z: i[3],L1:X0[0][0],\
                    L2:X0[1][0],L3:X0[2][0],L4:X0[3][0],L5:X0[4][0],\
                    L6:X0[5][0],L7:X0[6][0],L8:X0[7][0],L9:X0[8][0],\
                    L10:X0[9][0],L11:X0[10][0]})
        val_L0=np.append(val_L0,val)     
L0=np.matrix(val_L0)
L0=np.transpose(L0)

# Vetor das observações Lb-----------------------------------------------------
Lb=np.matrix(xy_ref_foto_pc)
Lb=np.transpose(Lb)

#Matriz Peso-------------------------------------------------------------------
I=np.identity(12, dtype=float)
P=I*(1/((tp)**2))
P=np.matrix(P)

# Iteração----------------------------------------------------------------------
stop = False
limite_iter = 10**(-6)
it = 1
while not stop:
    if it != 1:
        X0 = np.array(Xa)

    #Matriz dos parâmetros aproximados (L0)------------------------------------
    val_X0=[]
    for i in pa_pc:
        for j in [xp,yp]:
            val= j.evalf(subs={X: i[1], Y: i[2],\
                    Z: i[3],L1:X0[0][0],\
                    L2:X0[1][0],L3:X0[2][0],L4:X0[3][0],L5:X0[4][0],\
                    L6:X0[5][0],L7:X0[6][0],L8:X0[7][0],L9:X0[8][0],\
                    L10:X0[9][0],L11:X0[10][0]})
            
            val_X0=np.append(val_X0,val)
            L0=np.matrix(val_X0)
            L0=np.transpose(L0)
            
    #Vetor L------------------------------------------------------------------
    L=L0-Lb
    
    #Matriz A------------------------------------------------------------------
    A=matrizA(pa_pc,xp,yp,X0)    
    At = np.transpose(A)
    N=(At*P*A).astype(np.float)
    Uu=(At*P*L).astype(np.float)
    
    #Vetor correção------------------------------------------------------------
    X_c =-inv(N)*Uu
    Xa=X0+X_c
    
    #residuo-------------------------------------------------------------------
    V =A*X_c+L
    
    #Sigma posteriori----------------------------------------------------------
    Vt=np.transpose(V)
    sigma_pos=Vt*P*V/(12-6)
    
    #MCV dos Parâmetros--------------------------------------------------------
    mvc_p=np.array(pinv(N))
    
    #MVC dos valores ajustados-------------------------------------------------
    mvc_vaj=A*pinv(N)*At
    
    if it != 1: #Stop criteria
        if ((np.amax(abs(X_c))) <= limite_iter) or (it ==100):
            stop = True
        else:
            X0 = Xa
    it = it+1
    
    
np.set_printoptions(precision=4)

print()
print()
print('------------------------------------')
print('Resíduos (V)')
print()
for i in V.tolist():
    print(round(i[0], 4))
    

print()
print()
print('------------------------------------')
print('Parâmetros (L1,L2...L11)')
print()
print(Xa)

# DLT na sua forma inversa ----------------------------------------------------
# Obtenção de X,Y (espaço objeto) dos pv por meio de  Xa (L1,L2. L3...L11), 
# x,y (objeto imagem) e Z dos pv ----------------------------------------------
def DLT_inv(xa,xy_ref_foto_pv,pa_pv):
    xa=np.array(Xa)
    d=np.column_stack([xy_ref_foto_pv,pa_pv])
    XX=[]
    for k in d:
        a11=xa[0][0]-k[0]*xa[8][0]
        a12=xa[1][0]-k[0]*xa[9][0]
        a21=xa[4][0]-k[1]*xa[8][0]
        a22=xa[5][0]-k[1]*xa[9][0]
        c11=-k[5]*(xa[2][0]-k[0]*xa[10][0])-xa[3][0]+k[0]
        c12=-k[5]*(xa[6][0]-k[1]*xa[10][0])-xa[7][0]+k[1]
        AA=np.matrix([[a11, a12], [a21, a22]])
        C=np.matrix([[c11], [c12]])
        X=np.dot(inv(AA),C)
        XX=np.append(XX,X)
    return XX

# pa_pv em vetor---------------------------------------------------------------
pa_pv_vetor=[]
for i in pa_pv:
    X=i[1]
    Y=i[2]
    pa_pv_vetor=np.append(pa_pv_vetor,np.array([X,Y]))

pa_pv_cal=DLT_inv(Xa,xy_ref_foto_pv,pa_pv)

# Diferença entre pv calculados e observados no espaço objeto------------------
pa_pv_dif=pa_pv_cal-pa_pv_vetor
print()
print()
print('------------------------------------')
print('Diferença entre pv calculados e observados no espaço objeto')
print()
print(pa_pv_dif)

print()
print()
print('------------------------------------')
print('Erro Médio Quadrático')
erro=[]
erro_total=[]
for i in range(0,len(pa_pv_cal),1):
    if i % 2 == 0: #par
        x_erro=pa_pv_dif[i]
        y_erro=pa_pv_dif[i+1]
        erro=np.append(erro,(x_erro**2+y_erro**2)**(1/2))
        erro_total=np.append(erro_total,(x_erro**2+y_erro**2)/6)
print(erro)
print()
print('Erro Total')
erro_total=(sum(erro_total))**(1/2)
print(erro_total)


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


    
