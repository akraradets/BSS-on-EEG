import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
class Filter:
    TYPE_BUTTERWORTH = 0
    TYPE_CHEBYSHEV1 = 1
    TYPE_CHEBYSHEV2 = 2
    TYPE_ELLIPTICOL = 3

    LIST_TYPE = {
        0: "BUTTERWORTH",
        1: "CHEBYSHEV1",
        2: "CHEBYSHEV2",
        3: "ELLIPTICOL"
    }

    def __init__(self):
        self.type = Filter.TYPE_BUTTERWORTH
        self.order = 2
        self.wn = -1
        self.fs = -1
        self.rp = 2 
        self.rs = 20
        self.btype = ""
        
        self.ba = None

    @staticmethod
    def getFilterType(enum):
        return Filter.LIST_TYPE.get(enum, "Invalid numbner")

    def setLowCut(self, lowcut):
        self.wn = lowcut
        self.btype = 'highpass'

    def setHighCut(self, highcut):
        self.wn = highcut
        self.btype = 'lowpass'
    
    def setBandPass(self, lowcut, highcut=-1):
        if(type(lowcut) == type(tuple())):
            self.wn = [lowcut[0],lowcut[1]]
        else:
            if(highcut == -1): raise ValueError(f"You either set both lowcut and highcut or use lowcut with tuple (lowcut,highcut)")
            self.wn = [lowcut,highcut]
        self.btype = 'bandpass'

    def setFS(self,fs):
        self.fs = fs

    def setFilterType(self, type):
        if(type not in Filter.LIST_TYPE.keys()): raise ValueError(f"Filter Type should be set using Filter.TYPE_...")
        self.type = type

    def loadFilter(self, order=2):
        if(self.wn == -1): raise ValueError(f"set 'lowcut', 'highcut' or 'bandpass' first")
        if(self.btype == ""): raise ValueError(f"Somehow the developer forget to write 'btype'")
        if(self.fs == -1): raise ValueError(f"set 'fs' first") 

        self.order = order
        b,a = -1,-1
        if(self.type == Filter.TYPE_BUTTERWORTH):
            b,a = signal.butter(N=self.order, Wn=self.wn,btype=self.btype,output='ba',fs=self.fs) 
        
        elif(self.type == Filter.TYPE_CHEBYSHEV1):
            b,a = signal.cheby1(N=self.order, Wn=self.wn,btype=self.btype,output='ba',fs=self.fs, rp=self.rp) 
        
        elif(self.type == Filter.TYPE_CHEBYSHEV2):
            b,a = signal.cheby2(N=self.order, Wn=self.wn,btype=self.btype,output='ba',fs=self.fs, rs=self.rs) 
        
        elif(self.type == Filter.TYPE_ELLIPTICOL):
            b,a = signal.ellip(N=self.order, Wn=self.wn,btype=self.btype,output='ba',fs=self.fs, rp=self.rp, rs=self.rs) 

        else:
            raise ValueError(f"Unknow Filter Type. {Filter.LIST_TYPE}")            

        self.ba=(b,a)

    def getFreqencyResponse(self):
        if(self.ba == None): raise ValueError(f"You must loadFilter first")

        b,a = self.ba
        w,n = signal.freqz(b,a,worN=(self.fs//2))
        w = w/np.pi
        n = abs(n)
        return w,n

    def plotFreqencyResponse(self):
        w,n = self.getFreqencyResponse()
        plt.plot(w,n)
        plt.show()

    def applyFilter(self, s):
        b,a = self.ba
        filtered_s = signal.lfilter(b,a,s)
        return filtered_s