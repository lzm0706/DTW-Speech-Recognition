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
#将文件名中的\转换以免出现转义字符造成错误
str1=sys.argv[1]
str2=sys.argv[2]
trainlist=str1.replace('\\','/')
model=str2.replace('\\','/')
print('---opening txt files---\n')
trainlistob=open(trainlist)
modelob=open(model,'w')

for line in trainlistob.readlines():
    #获得训练文件名
    label=line.split(' ')[0]
    name=re.sub('\n','',line.split(' ')[1])
    #print(name)
    filename=name.replace('\\','/')
    #print(filename)
    #文件读取
    wavearr=wavread(filename)
    #信号预加重
    wavearr_pre=pre_emphasis(wavearr,0.98)
    #选择窗函数
    winfunc = signal.hamming(240)
    #信号分帧
    n=enframe(wavearr_pre,240,80,winfunc)
    #vad
    na=vad.vioceextrac(n)
    #提取特征
    feat=feature.mfcc(na,512)
    numpy.set_printoptions(threshold=numpy.nan,linewidth=numpy.nan)
    #将特征和标准保存到输出的模型文件
    print('---output results of',name,'---\n')
    print(label,file=modelob)
    print(feat,file=modelob)
    #print(numpy.shape(feat))
print('\n---training complete---\n')    
trainlistob.close
modelob.close


