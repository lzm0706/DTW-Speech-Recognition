import sys
import re
from wav import wavread
from pre import pre_emphasis
from frame import enframe
import scipy.signal as signal
import numpy
import vad
import feature

str1=sys.argv[1]
str2=sys.argv[2]
trainlist=str1.replace('\\','/')
model=str2.replace('\\','/')

trainlistob=open(trainlist)
modelob=open(model,'w')

for line in trainlistob.readlines():
    label=line.split(' ')[0]
    name=re.sub('\n','',line.split(' ')[1])
    #print(name)
    filename=name.replace('\\','/')
    #print(filename)
    wavearr=wavread(filename)

    wavearr_pre=pre_emphasis(wavearr,0.98)

    winfunc = signal.hamming(240)
    n=enframe(wavearr_pre,240,80,winfunc)


    na=vad.vioceextrac(n)
    
    feat=feature.mfcc(na,512)
    numpy.set_printoptions(threshold=numpy.nan,linewidth=numpy.nan)
    print(label,file=modelob)
    print(feat,file=modelob)
    #print(numpy.shape(feat))
    
trainlistob.close
modelob.close


