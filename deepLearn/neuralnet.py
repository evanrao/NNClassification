from __future__ import division
import numpy as np
import math,os,formula

class Neural:
    actfc =  'tanh_opt'
    lrate = 2
    outfc = 'sigm'
    arrln = 0
    dpf = 0
    addinval = 0
    momentum = 0.5

    arr,wg,wgv,avac,dwg = [],[],[],[[]],[]
    odata,er,lo,dom = [],[],[],[]

    def __init__(self,arrin):
        print('I\'m Neural,Init now!')

        self.arr = arrin
        self.arrln = len(self.arr)

        if self.arrln < 3:
            print 'Error::The one-dimensional matrix length must be greater than three '
            os._exit(0)

        for i in range(1,self.arrln):
            self.wg.append(  (np.random.rand(self.arr[i],self.arr[i-1]+1) -0.5)  * 8 * math.sqrt( 6/ (self.arr[i] + self.arr[i-1]) ) )
            self.wgv.append(np.zeros((self.arr[i],self.arr[i-1]+1)))
            self.avac.append(np.zeros((1,self.arr[i])))
            self.dwg.append([])

