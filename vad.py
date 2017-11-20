import numpy as np
import pylab as pl

def sgn(n):
    if n>=0:
        return 1
    if n<0:
        return -1

def energy(frames):
    #print(frames)
    frames=frames/np.amax(np.absolute(frames))#归为[-1,1]
    nframe=np.shape(frames)[0]
    lframe=np.shape(frames)[1]
    energy=np.power(frames,2)
    e=energy.sum(axis=1)
    time = np.arange(0, nframe)
    pl.plot(time,e)
    #pl.show()
    return e

def zcr(frames):
    nframe=np.shape(frames)[0]
    lframe=np.shape(frames)[1]
    zcr=np.zeros(nframe)
    for i in range(nframe):
        zframe=0
        for j in range(1,lframe):
            zframe=zframe+0.5*(abs(sgn(frames[i,j])-sgn(frames[i,j-1])))
        zcr[i]=zframe/(lframe-1)
    time = np.arange(0, nframe)
    pl.plot(time,zcr)
    #pl.show()
    return zcr

def vioceextrac(frames):
    print('---extracting active voice----\n')
    nframe=np.shape(frames)[0]
    lframe=np.shape(frames)[1]
    mh=10
    ml=2
    zs=0.18
    a1=0
    a2=0
    status=0
    count=0
    e=energy(frames)
    z=zcr(frames)
    mh=min(mh,np.amax(e)/4)
    ml=min(ml,np.amax(e)/8)
    #print(mh,ml)
    for i in range(nframe):
        if status==0 and e[i]>mh:
            a1=i
            status=1
        if status==1 and e[i]<mh:
            a2t=i
            status=2
        if status==2 and e[i]<mh:
            count=count+1
        if status==2 and e[i]>mh:
            count=0
            status=1
        if status==2 and count>30:
            a2=a2t
    #print(a1,a2)
    b1=a1-1
    b2=a2+1
    while e[b1]>ml:
        b1=b1-1
    while e[b2]>ml:
        b2=b2+1
    #print(b1,b2)
    c1=b1-1
    c2=b2+1
    while z[c1]>=(3*zs):
        c1=c1-1
    while z[c2]>=(3*zs):
        c2=c2+1
    #print(c1,c2)
    frames_a=frames[c1:c2,:]
    return(frames_a)