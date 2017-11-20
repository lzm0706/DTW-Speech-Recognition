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
#将文件名中的\转换以免出现转义字符造成错误
model=str1.replace('\\','/')
testlist=str2.replace('\\','/')
result=str3.replace('\\','/')

testlistob=open(testlist)

resultob=open(result,'w')


for line in testlistob.readlines():
    #获得测试文件名
    name=re.sub('\n','',line)
    #将文件名中的\转换以免出现转义字符造成错误
    filename=name.replace('\\','/')
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
    feattest=feature.mfcc(na,512)
    #print(numpy.shape(feattest))
    
    '''
    读取模型文件
    '''
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
            #重新生成numpy矩阵并统一规格
            feat=numpy.array(featarr,dtype=numpy.float64)
            feattrain=feat.reshape(nf,m)
            #print(feattrain)
            featarr=[]
            #计算测试信号到模型中每个信号的dtw距离
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
            #重新生成numpy矩阵并统一规格
            feat=numpy.array(featarr,dtype=numpy.float64)
            feattrain=feat.reshape(nf,m)
            #print(feattrain)
            featarr=[]
            #计算测试信号到模型中每个信号的dtw距离
            dist.append(sr.score(feattrain,feattest))
            #print(nf,m)
    print('---min distance matching---\n')
    #在所有距离中搜索最小的
    labelnum=dist.index(min(dist))
    print('---output matching result---\n')
    #输出最小距离信号对应的label作为识别结果
    print(label[labelnum],file=resultob)
    modelob.close
print('\n---recognition complete---\n')
            
testlistob.close
resultob.close
