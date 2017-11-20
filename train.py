import sys
import re
from wav import wavread
from pre import pre_emphasis
from frame import enframe
import scipy.signal as signal
import numpy
import vad
import feature
print('\n---get command---\n')
str1=sys.argv[1]
str2=sys.argv[2]
trainlist=str1.replace('\\','/')
model=str2.replace('\\','/')
print('---opening txt files---\n')
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
    print('---output results of',name,'---\n')
    print(label,file=modelob)
    print(feat,file=modelob)
    #print(numpy.shape(feat))
print('\n---training complete---\n')    
trainlistob.close
modelob.close


