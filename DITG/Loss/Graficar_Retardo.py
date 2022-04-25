import subprocess
import matplotlib.pyplot as pl
import numpy as np

pl.figure()
pl.plot([0,1,2,3,4],[0.207029,0.0053,0.005224,0.005,0.00558])
pl.xlabel('Link Packet Loss [%]')
pl.ylabel('Delay [ms]')
pl.savefig('Delay_VS_Loss.png',format='png')
