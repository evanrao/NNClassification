from __future__ import division
import numpy as np
from scipy import io


def dataload(filename):
    data =  np.load(filename)
    #print(data)
    return data

# Here, the data is normalized between 0 and 1 by using the logarithmic process of 10 for the sample
def log_normalization(data):
    colnum = len(data[0])
    rownum = len(data)
    #print data[0:len(data),0]
    nordata = np.zeros((rownum, colnum))
    for i in range(0, colnum):
        tempi = np.log10(data[0:rownum,i])
        tempi = (tempi - np.min(tempi)) / (np.max(tempi) - np.min(tempi))
        nordata[0:rownum,i] = tempi
    return nordata

def normalization(data):
    colnum = len(data[0])
    rownum = len(data)
    nordata = np.zeros((rownum, colnum))
    for i in range(0, colnum):
        nordata[0:rownum,i] = (data[0:rownum,i] - np.min(data[0:rownum,i])) / (np.max(data[0:rownum,i]) - np.min(data[0:rownum,i]))
    return nordata

#The mat data files are converted to npy files
def mattonpz(srcname,srckey,outname):
    in_d = io.loadmat(srcname)
    oct_a = in_d[srckey]
    oct_a = float(oct_a)
    np.save(outname,oct_a)

if __name__ == '__main__':
    print log_normalization(dataload('in_data.npy'))
