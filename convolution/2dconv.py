import numpy as np 
from numpy.lib.stride_tricks import as_strided
import time

X = np.random.randint(1,9,size=(10,10))
K = np.full((3,3),10)

Xr, Xc = X.shape
Kr, Kc = K.shape
u = np.array(X.itemsize)
Xstrided = as_strided(X, shape=(Xr-Kr+1, Xc-Kc+1, Kr, Kc), strides=u*(Xc,1,Xc,1))

R = np.full(Xstrided.shape[:2], np.nan)
Rr, Rc = R.shape

highlight = np.vectorize(lambda x: f"\x1b[33m10\x1b[0m")
for i, row in enumerate(Xstrided):
    for j, window in enumerate(row):
        temp = window.copy()
        window[...] = K
        R[i,j] = np.tensordot(K,temp)
        Xstr = str(X).replace('10 10 10','\x1b[30;43m10 10 10\x1b[0m')
        Rstr = str(R).replace('nan','   ')
        print(f"{Xstr}\n\n{Rstr}\x1b[{Xr+Rr+1}A")
        window[...] = temp
        time.sleep(0.05)

print(f"\x1b[{Xr+Rr}B")

