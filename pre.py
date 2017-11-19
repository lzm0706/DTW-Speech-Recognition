import numpy
def pre_emphasis(signal,coefficient=0.95):
    '''
    对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])