import wave  
import numpy as np
import pylab as pl

def wavread(filename):
    #读取wav文件
    print('---reading wav file  ',filename,'---\n')
    wavefile = wave.open(filename, 'r')

    #读取wav文件的四种信息的函数。numframes表示一共读取了几个frames
    nchannels = wavefile.getnchannels()  
    sample_width = wavefile.getsampwidth()  
    framerate = wavefile.getframerate()  
    numframes = wavefile.getnframes()  

    #print("channel",nchannels)  
    #print("sample_width",sample_width)  
    #print("framerate",framerate)  
    #print("numframes",numframes)

    str_data = wavefile.readframes(numframes)
    wavefile.close()

    #将波形数据转换为数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    #print(len(wave_data))
    time = np.arange(0, numframes) * (1.0 / framerate)
    #print(len(time))
    '''
    # 绘制波形
    pl.plot(time, wave_data)
    pl.xlabel("time (seconds)")
    pl.show()
    '''
    return wave_data