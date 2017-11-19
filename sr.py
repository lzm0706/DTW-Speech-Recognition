import numpy as np

def dist(feat1,feat2):
    n=np.shape(feat1)[0]
    m=np.shape(feat2)[0]
    d=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            d[i,j]=np.sqrt(np.sum(np.square(feat1[i,:]-feat2[j,:])))
    return d
    
def dtw(dist):
    realmax=1.79E+308
    n=np.shape(dist)[0]
    m=np.shape(dist)[1]
    D=np.ones((n+1,m+1))*realmax
    D[0,0]=0
    for i in range(1,n+1): 
        for j in range(1,m+1):
            D[i,j]=dist[i-1,j-1]+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[n,m]
    
def score(feat1,feat2):
    return dtw(dist(feat1,feat2))