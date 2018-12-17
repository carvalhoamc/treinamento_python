#introducao ao numpy

import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
print("shape =")
print(a.shape)
print("linha = ")
print(a.shape[0])
print("coluna = ")
print(a.shape[1])
print("nome do tipo = ")
print(a.dtype.name)
print("tamanho do item = ")
print(a.itemsize)
print("size = ")
print(a.size)
print("tipo = ")
print(type(a))

#Criando um array numpy
a = np.array([2,3,4])
print(a)
b = np.array([(1.5,2,3), (4,5,6)])
print(b)
c = np.array( [ [1,2], [3,4] ], dtype=complex )
print(c)
print(c.dtype)
#criando matriz de zeros
x = np.zeros( (3,4) )
print(x)
#criando matriz de 1s
y = np.ones( (2,3,4), dtype=np.int16 )
print(y)
#criando valores em sequencia (inteiros)
sequencia = np.arange( 10, 30, 3 )
print(sequencia)
sequencia1 = np.arange( 0, 2, 0.3 )
print(sequencia1)
#criando valores em sequencia (float)
sequencia2 = np.linspace( 0, 2, 9 )
print(sequencia2)
#criando numeros aleatorios
aleatorio = np.random.rand(3,2)
print(aleatorio)
#mudando o formato do array
a = np.arange(6)                         # 1d array
print('1D array')
print(a)
b = np.arange(12).reshape(4,3)           # 2d array
print('2D array')
print(b)
c = np.arange(24).reshape(2,3,4)         # 3d array
print('3D array')
print(c)
#forca impressao do array completo
np.set_printoptions(threshold=np.nan)
print(np.arange(100))
print(np.arange(100).reshape(10,10))

#operacoes basicas
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print(a)
print(b)
c = a - b
print(c)
f = b**2
g = 10*np.sin(a)

if np.mean(f) > np.mean(g):
    print('f > g')
    print(f)
else:
    print('f < g')
    print(g)

A = np.array( [[1,1],
            [0,1]] )
B = np.array( [[2,0],
            [3,4]] )

print('A * B = ')
print(A * B)
print('A @ B = ')
print(A @ B)
print('A.dot(B)')
print(A.dot(B))

a = np.ones((2,3), dtype=int)
print(a)
b = np.random.random((2,3))
print(b)
a *= 3
print(a)
b += a
print(b)
a += b.astype('int64') #retire o cast! Veja o que acontece.
print(a)

a = np.random.random((2,3))
print(a)
print(a.sum())
print(a.min())
print(a.max())

b = np.arange(12).reshape(3,4)
print(b)
print(b.sum(axis=0) )
print(b.min(axis=1))
print(b.cumsum(axis=1))

#Indexing, Slicing and Iterating¶

a = np.arange(10)**3
print(a)
print(a[2])
print(a[2:5])
a[:6:2] = -1000
print(a)
print(a[::-1])

for i in a:
    print(i**(1/3.))

