#funcoes
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt

def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)
print(b)


print(b[2,3])
print(b[0:5, 1])              # each row in the second column of b
print(b[ : ,1])               # equivalent to the previous example
print(b[1:3, : ])             # each column in the second and third row of b
print(b[-1])

c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])

print(c[1,...])
print(c[...,2])

for element in b.flat:
    print(element)


#shape manipulation
a = np.floor(10*np.random.random((3,4)))
print(a)
a.ravel()  # returns the array, flattened
print(a)
a.reshape(6,2)  # returns the array with a modified shape
print(a)
print(a.T)
print(a.T.shape)

aa = a.resize((2,6))
print(aa)

a = a.reshape(3,-1)
print(a)

a = np.floor(10*np.random.random((2,2)))
print(a)
b = np.floor(10*np.random.random((2,2)))
print(b)
vs = np.vstack((a,b))
print(vs)
hs = np.hstack((a,b))
print(hs)

v2d = np.column_stack((a,b))     # with 2D arrays


a = np.array([4.,2.])
b = np.array([3.,8.])
ab = np.column_stack((a,b))     # returns a 2D array
ba = np.hstack((a,b))           # the result is different

print(a[:,newaxis])               # this allows to have a 2D columns vector


print(np.column_stack((a[:,newaxis],b[:,newaxis])))
print(np.hstack((a[:,newaxis],b[:,newaxis])))   # the result is the same


a = np.floor(10*np.random.random((2,12)))
print(a)
aa = np.hsplit(a,3)   # Split a into 3
print(aa)

bb = np.hsplit(a,(3,4))   # Split a after the third and the fourth column
print(bb)

a = np.arange(12)
b = a            # no new object is created
print(b is a)           # a and b are two names for the same ndarray object

b.shape = 3,4    # changes the shape of a
print(b.shape)
a.shape
print(a.shape)


c = a.view()
print(c)
print(c is a)
print(c.base is a )                        # c is a view of the data owned by a
print(c.flags.owndata)

s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
print(s)
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
print(s)
print(a)


d = a.copy()                          # a new array object with new data is created
print(d)
print(d is a)

print(d.base is a)                           # d doesn't share anything with a

d[0,0] = 9999
print(d)

#indexing
a = np.arange(12).reshape(3,4)
b = a > 4
print(b)                                          # b is a boolean with a's shape
print(a[b])

def mandelbrot( h,w, maxit=20 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime

plt.imshow(mandelbrot(400,400))
plt.show()

#histograma
import numpy as np
import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)
plt.show()