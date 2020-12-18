
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


# Cálculo da média de X,Y,Z----------------------------------------------------
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

def CL_to_xp(tpx,tpy,coluna,linha,U,T,xi,yi,k1,k2,k3):
    # Convertendo para o referencial com origem no centro da imagem (CL-->xl,yl)
    xl = tpx * (coluna - (U-1)/2)
    yl = -tpy * (linha - (T-1)/2)
    # Coordenadas em relação ao pp---------------------------------------------
    xpp= xl - xi
    ypp= yl - yi
    #Distorção radial simétrica------------------------------------------------
    r = mt.sqrt(xpp**2 + ypp**2)
    drsx = xpp * (k1*r**2 + k2*r**4 + k3*r**6)
    drsy = ypp * (k1*r**2 + k2*r**4 + k3*r**6)
    #Distorção descentrada-----------------------------------------------------
    ddx = p1*(r**2 + 2*xpp**2) + 2*p2*xpp*ypp
    ddy = 2*p1*xpp*ypp + p2*(r**2 + 2*ypp**2)
    # Coordenadas corrigidas---------------------------------------------------
    x = xpp - drsx - ddx
    y = ypp - drsy - ddy
    return x,y

#Modelo funcional: Equação de Colinearidade Direta----------------------------
    
X, Y, Z, X0, Y0, Z0, om, fi, kapa, c  = symbols('X Y Z X0 Y0 Z0 om fi kapa c')

m11 = cos(fi) * cos(kapa)
m12 = cos(om) *sin(kapa) + sin(om)*sin(fi)*cos(kapa)
m13 = sin(om) *sin(kapa) - cos(om)*sin(fi)*cos(kapa)

m21 = -cos(fi)* sin(kapa)
m22 = cos(om)*cos(kapa) - sin(om)*sin(fi)*sin(kapa)
m23 = sin(om)*cos(kapa) + cos(om)*sin(fi)*sin(kapa)

m31 = sin(fi)
m32 = -sin(om)*cos(fi)
m33 = cos(om)*cos(fi)

x = - c * (m11*(X-X0) + m12*(Y-Y0) + m13*(Z-Z0)) / (m31*(X-X0) + m32*(Y-Y0) + m33*(Z-Z0))
y = - c * (m21*(X-X0) + m22*(Y-Y0) + m23*(Z-Z0)) / (m31*(X-X0) + m32*(Y-Y0) + m33*(Z-Z0))


# Vetor - POEs aproximados iniciais--------------------------------------------
x0= np.array([[Xm],[Ym],[hvoo+Zm],[0.001],[0.001],[1]]) 

# Montar a matriz A (12x6)-----------------------------------------------------
def matrizA(x0,pa_pc,x,y):  
    a1 = diff(x,X0)
    a2 = diff(x,Y0)
    a3 = diff(x,Z0)
    a4 = diff(x,om)
    a5 = diff(x,fi)
    a6 = diff(x,kapa)
    
    a7 = diff(y,X0)
    a8 = diff(y,Y0)
    a9 = diff(y,Z0)
    a10 = diff(y,om)
    a11= diff(y,fi)
    a12 = diff(y,kapa)
    
    coef_x = [a1, a2, a3, a4, a5, a6]
    coef_y = [a7, a8, a9, a10, a11, a12]
    
    def IterA(coef,pa_pc,x0):
        val_A=[]
        for i in pa_pc:
            for j in coef:
                num_a = j.evalf(subs={c: (15.8893), X0: x0[0][0], Y0: x0[1][0],\
                                      Z0: x0[2][0],om: x0[3][0], fi: x0[4][0],\
                                      kapa: x0[5][0], X:i[1],Y:i[2],Z:i[3]})
                val_A=np.append(val_A,num_a)
        return val_A       
    
    val_Ax=IterA(coef_x,pa_pc,x0)
    val_Ay=IterA(coef_y,pa_pc,x0)
    A=np.matrix([val_Ax[0:6], val_Ay[0:6], val_Ax[6:12], val_Ay[6:12],\
                val_Ax[12:18], val_Ay[12:18],val_Ax[18:24],val_Ay[18:24],\
                val_Ax[24:30], val_Ay[24:30], val_Ax[30:36], val_Ay[30:36]])
    return A

# Vetor das observações Lb-----------------------------------------------------
Lb=[]
for i in cl_pc:
    xx,yy=CL_to_xp(tpx,tpy,i[0],i[1],U,T,xi,yi,k1,k2,k3)
    Lb=np.append(Lb,np.array([xx,yy]))
Lb=np.matrix(Lb)
Lb=np.transpose(Lb)

# Determinação de x,y dos pontos de verificação a partir de CL-----------------
cl_pv_to_xy=[]
cl_pv_to_x=[]
cl_pv_to_y=[]
for i in cl_pv:
    xx,yy=CL_to_xp(tpx,tpy,i[0],i[1],U,T,xi,yi,k1,k2,k3)
    cl_pv_to_x=np.append(cl_pv_to_x,np.array([xx]))
    cl_pv_to_y=np.append(cl_pv_to_y,np.array([yy]))
    cl_pv_to_xy=np.append(cl_pv_to_xy,np.array([xx,yy]))

#Matriz Peso-------------------------------------------------------------------
I=np.identity(12, dtype=float)
P=I*(1/((tp)**2))
P=np.matrix(P)
#P=I
# Iteração---------------------------------------------------------------------
stop = False
limite_inter = 10**(-6)
it = 1
while not stop:
    if it != 1:
        x0 = np.array(Xa)
    print("Iteração: ", it)
   
    #Matriz dos parâmetros aproximados (L0)------------------------------------
    L_0=[]
    for i in pa_pc:
        for j in [x,y]:
            xii= j.evalf(subs={c: (15.8893), X0: x0[0][0], Y0: x0[1][0], Z0: x0[2][0],\
                              om: x0[3][0], fi: x0[4][0], kapa: x0[5][0], X:i[1],Y:i[2],Z:i[3]})
            L_0=np.append(L_0,xii)
        L00=np.matrix(L_0)
        L0=np.transpose(L00)
        
    #Matriz L------------------------------------------------------------------
    L=L0-Lb
    
    #Matriz A------------------------------------------------------------------
    A=matrizA(x0,pa_pc,x,y).astype(np.float)       
    At = np.transpose(A)
    part1 = np.dot(At, P);  
    N= np.dot(part1, A).astype(np.float)
    Uu = np.dot(part1,L).astype(np.float)
    
    # Vetor correção-----------------------------------------------------------
    X_c = -np.dot(inv(N),Uu)
    
    # Vetor dos parâmetros ajustados-------------------------------------------
    Xa=x0+X_c
    
    # Matriz dos resíduos------------------------------------------------------
    V = np.dot(A, X_c) + L
    
    # Variância a posteriori---------------------------------------------------
    Vt=np.transpose(V)
    sigma_pos=Vt*P*V/(12-6)
    
    # MCV dos Parâmetros-------------------------------------------------------
    mvc_p=np.array(inv(N))
    
    #MVC dos Valores Ajustados-------------------------------------------------
    mvc_vaj=np.array(A*inv(N)*At)
            
    #print(np.amax(abs(X_c)))
    
    if it != 1: 
        if ((np.amax(abs(X_c))) <= limite_inter) or (it ==100):
            stop = True
        else:
            x0 = Xa    
    it = it+1
    

print()
print()
print('------------------------------------')
print('Resíduos (V)')
print()
for i in V.tolist():
    print(round(i[0], 4))
    
np.set_printoptions(precision=4)

print()
print('Vetor dos parâmetros ajustados - POEs ajustados')
print()
print(Xa)

Xa=np.array(Xa)

# Equações de colinearidade inversa--------------------------------------------
# Determninação de X, Y (espaço objeto) a partir dos POEs ajustados e x,y (espaço imagem) e Z
def Col_inv_sub(Xa,cl_pv_to_x,cl_pv_to_y,pa_pv):
    X, Y, Z, X0, Y0, Z0, om, fi, kapa, c, x, y  = symbols('X Y Z X0 Y0 Z0 om fi kapa c, x, y')
    
    x_inv = (Z-Z0)*((m11*(x) + m21*(y) + m31*(-c))/(m13*(x) + m23*y + m33*(-c)))+X0
    y_inv = (Z-Z0)*((m12*(x) + m22*(y) + m32*(-c))/(m13*(x) + m23*(y) + m33*(-c)))+Y0
    dd=np.column_stack([cl_pv_to_x,cl_pv_to_y,pa_pv])
    
    val_AA=[]
    for i in dd:
        for l in [x_inv,y_inv]:
            num_aa= l.evalf(subs={c: (15.8893), X0: Xa[0][0], Y0: Xa[1][0], Z0: Xa[2][0],\
                          om: Xa[3][0], fi: Xa[4][0], kapa: Xa[5][0], x: i[0], y: i[1], Z: i[5]})
            val_AA=np.append(val_AA,num_aa)
    return val_AA       

pa_pv_to_xy=[]
for i in pa_pv:
    pa_pv_to_xy=np.append(pa_pv_to_xy,np.array([i[1],i[2]]))

xy_inv=Col_inv_sub(Xa,cl_pv_to_x,cl_pv_to_y,pa_pv)       

# Diferença entre os pontos de verificação calculados e observados no espaço objeto
print()
print()
print('Diferença entre pv calculados e observados no espaço objeto')
print()
dif_entre_pv_objeto=pa_pv_to_xy-xy_inv
print(dif_entre_pv_objeto)

##################3

print()
print('------------------------------------')
errox=[]
erroy=[]
erro_total=[]
for i in range(0,len(dif_entre_pv_objeto),1):
    if i % 2 == 0: #par
        x_erro=dif_entre_pv_objeto[i]
        y_erro=dif_entre_pv_objeto[i+1]
        errox=np.append(errox,((x_erro**2)))
        erroy=np.append(erroy,((y_erro**2)))
        erro_total=np.append(erro_total,((x_erro**2+y_erro**2)))
        
errox=(sum(errox)/6)**(1/2)
erroy=(sum(erroy)/6)**(1/2)
print('Erro X')
print(errox)
print('Erro Y')
print(erroy)
print()
print('Erro Total (X,Y)')
erro_total=(sum(erro_total)/6)**(1/2)
print(erro_total)


# Análise da qualidade do ajustamento (qui-quadrado para alfa = 5%)-----------
Qui_cal=Vt*P*V
Qui_tab_025=1.237
Qui_tab_975=14.449

print()
print()
print('Qui-quadrado para alfa = 5%')
print()
print(Qui_tab_025,'<',float(Qui_cal),'<', Qui_tab_975)

# Análise das precisões--------------------------------------------------------
#As raízes quadradas dos elementos da diagonal principal da matriz MVC das 
#observações ajustadas  irão fornecer a precisão das observações ajustadas ----
diag_mvc_p =np.array([row[i] for i,row in enumerate(mvc_p)])
presi=(diag_mvc_p)**(1/2)
print()
print()
print('Precisão das Parâmetros ajustadas')
print()
print(presi)


# Análise das precisões--------------------------------------------------------
#As raízes quadradas dos elementos da diagonal principal da matriz MVC das 
#observações ajustadas  irão fornecer a precisão das observações ajustadas ----
diag_mvc_p =np.array([row[i] for i,row in enumerate(mvc_p)])
presi=(diag_mvc_p)**(1/2)
print()
print()
print('Precisão das observações ajustadas')
print()
print(presi)

print()
print()
print('Valor T calculado > Valor T tabelado (alfa = 5%)')
print()
valor_t_tab=1.943
for i in range(0,len(Xa),1):
    valor_t_cal=abs(Xa[i][0])/((diag_mvc_p[i])**(1/2))
    print(valor_t_cal,'>',valor_t_tab)
    
print()
print()
print('Sigma Posteriori')
print()
print(sigma_pos)