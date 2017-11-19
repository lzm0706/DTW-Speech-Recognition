from wav import wavread
from pre import pre_emphasis
from frame import enframe
import scipy.signal as signal
import numpy
import vad
import pylab as pl
import feature
import sr
#numpy.set_printoptions(threshold=numpy.nan)

wavearr1=wavread("E:\\data\\Github\\DTW_based_Isolated_Word_Recognition\\data\\train\\0.wav")
wavearr2=wavread("E:\\data\\Github\\DTW_based_Isolated_Word_Recognition\\data\\test\\7.wav")

wavearr_pre1=pre_emphasis(wavearr1,0.98)
wavearr_pre2=pre_emphasis(wavearr2,0.98)

winfunc = signal.hamming(240)
n1=enframe(wavearr_pre1,240,80,winfunc)
n2=enframe(wavearr_pre2,240,80,winfunc)

n1=vad.vioceextrac(n1)
n2=vad.vioceextrac(n2)

feat1=feature.mfcc(n1,512)
feat2=feature.mfcc(n2,512)

print(sr.score(feat1,feat2))

