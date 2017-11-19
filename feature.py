import numpy
import scipy.io.wavfile
from scipy.fftpack import dct

def mfcc(frames,NFFT=512,sample_rate=16000):
    #帧能量谱
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    energy=numpy.sum(pow_frames,1)  #对每一帧的能量谱进行求和
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)  #对能量为0的地方调整为eps，这样便于进行对数处理
    #print(numpy.shape(mag_frames),numpy.shape(pow_frames))
    #滤波器组
    nfilt=40
    
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    
    #mfcc
    num_ceps = 13
    appendEnergy=1
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    if appendEnergy:
        mfcc[:,0]=numpy.log(energy)  #只取2-13个系数，第一个用能量的对数来代替
    #mean-normalized 
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    
    return mfcc_delta_delta(mfcc)
    
def derivate(feat,big_theta=2,cep_num=13):
    '''计算一阶系数或者加速系数的一般变换公式
    参数说明:
    feat:MFCC数组或者一阶系数数组
    big_theta:公式中的大theta，默认取2
    '''
    result=numpy.zeros(feat.shape) #结果
    denominator=0  #分母
    for theta in numpy.linspace(1,big_theta,big_theta):
        denominator=denominator+theta**2
    denominator=denominator*2 #计算得到分母的值
    for row in numpy.linspace(0,feat.shape[0]-1,feat.shape[0]):
        tmp=numpy.zeros((cep_num,))
        numerator=numpy.zeros((cep_num,)) #分子
        for t in numpy.linspace(1,cep_num,cep_num):
            a=0
            b=0
            s=0
            for theta in numpy.linspace(1,big_theta,big_theta):
                if (t+theta)>cep_num:
                    a=0
                else:
                    a=feat[int(row)][int(t+theta-1)]
                if (t-theta)<1:
                    b=0
                else:
                    b=feat[int(row)][int(t-theta-1)]
                s+=theta*(a-b)
            numerator[int(t-1)]=s
        tmp=numerator*1.0/denominator
        result[int(row)]=tmp
    return result  
    

def mfcc_delta(feat):
    '''计算13个MFCC+13个一阶微分系数
    '''
    result=derivate(feat) #调用derivate函数
    result=numpy.concatenate((feat,result),axis=1)
    return result 
    

def mfcc_delta_delta(feat):
    '''计算13个MFCC+13个一阶微分系数+13个加速系数,一共39个系数
    '''
    result1=derivate(feat)
    result2=derivate(result1)
    result3=numpy.concatenate((feat,result1),axis=1)
    result=numpy.concatenate((result3,result2),axis=1)
    return result