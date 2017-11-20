import sys
import re
from wav import wavread
from pre import pre_emphasis
from frame import enframe
import scipy.signal as signal
import numpy
import vad
import feature
import sr

str1=sys.argv[1]
str2=sys.argv[2]
str3=sys.argv[3]

model=str1.replace('\\','/')
testlist=str2.replace('\\','/')
result=str3.replace('\\','/')

testlistob=open(testlist)

resultob=open(result,'w')


for line in testlistob.readlines():
    name=re.sub('\n','',line)
    filename=name.replace('\\','/')
    wavearr=wavread(filename)

    wavearr_pre=pre_emphasis(wavearr,0.98)

    winfunc = signal.hamming(240)
    n=enframe(wavearr_pre,240,80,winfunc)

    na=vad.vioceextrac(n)
    
    feattest=feature.mfcc(na,512)
    #print(numpy.shape(feattest))
    modelob=open(model)
    print('---DTW---\n')
    i=0

    dist=[]
    label=[]
    mframes=modelob.readlines()
    #print(len(frames))
    for line1 in mframes:
        i+=1
        mline=line1.strip()
        if i==1 and len(mline)==1:
            label.append(mline)

            featarr=[]
            nf=0
        if i>1 and len(mline)==1:
            feat=numpy.array(featarr,dtype=numpy.float64)
            feattrain=feat.reshape(nf,m)
            #print(feattrain)
            featarr=[]
            dist.append(sr.score(feattrain,feattest))
            #print(nf,m)
            nf=0
            label.append(mline)

        if i!=len(mframes) and len(mline)>1:
            #print(mline)
            mline=mline.strip('[]')
            mline=mline.strip()
            #print(mline)
            mlinearr=re.split('\s+',mline)
            #print(len(mlinearr))
            #print(mlinearr)
            featarr.append(mlinearr)
            m=len(mlinearr)
            nf+=1
        if i==len(mframes) and len(mline)>1:
            #print(mline)
            mline=mline.strip('[]')
            mline=mline.strip()
            #print(mline)
            mlinearr=re.split('\s+',mline)
            #print(len(mlinearr))
            #print(mlinearr)
            featarr.append(mlinearr)
            m=len(mlinearr)
            nf+=1
            
            feat=numpy.array(featarr,dtype=numpy.float64)
            feattrain=feat.reshape(nf,m)
            #print(feattrain)
            featarr=[]
            dist.append(sr.score(feattrain,feattest))
            #print(nf,m)
    print('---min distance matching---\n')
    labelnum=dist.index(min(dist))
    print('---output matching result---\n')
    print(label[labelnum],file=resultob)
    modelob.close
print('\n---recognition complete---\n')
            
testlistob.close
resultob.close
