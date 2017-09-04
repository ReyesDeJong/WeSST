import pyedflib
import numpy as np
f = pyedflib.EdfReader("SC4001E0-PSG.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)
#%%

f2 = pyedflib.EdfReader("SC4001EC-Hypnogram.edf")
n2 = f2.signals_in_file
a=f2.readAnnotations()
#
#%%
f2._close()
del f2

f._close()
del f
#%%
#input single channel signal
def createWindowsBySamples(SingleChannelSignal, WSizeSamples):
    
    WSizeSamples=int(WSizeSamples)
    WindowsInSignal=int(SingleChannelSignal.size/WSizeSamples)
    ArrayOfWindows = np.zeros((WindowsInSignal, WSizeSamples))
    for i in np.arange(WindowsInSignal):
        ArrayOfWindows[i,:]=SingleChannelSignal[(i*WSizeSamples):((i+1)*WSizeSamples)]
    
    return ArrayOfWindows

T=0.01
WindowTime=30
WindowSamples=30/T

SCNSignal=sigbufs[0,:]

Windows=createWindowsBySamples(SCNSignal,WindowSamples)
        
        
    
#7950000=22.0833333hrs
#sigbufs2 = np.zeros((n2, f2.getNSamples()))
#T=0.01
#import main_load_edf
#
#psg_dir = 'SC4001E0-PSG.edf'
#ann_dir = 'SC4001EC-Hypnogram.edf'
#
#main_load_edf(psg_dir, ann_dir, "file")
#%%

