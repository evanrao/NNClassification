from __future__ import division
import time,os,ffbp
from neuralnet import Neural
import numpy as np
import formula as fa


validation = 0

xvar = 36
yvar = 6

nn = Neural(np.array([xvar, 100,yvar]))
for i in range(0, nn.arrln):
    nn.odata.append([])
    nn.dom.append([])

filesrc = np.load('./sdata/t36v6c6430s.npy')
filesi = len(filesrc)
insrc = filesrc[0:filesi, 0:xvar]
insrc = fa.zscore(insrc)
tgsrc = np.zeros((filesi, yvar))
for i in range(0, filesi):
    tgsrc[i, int(filesrc[i, xvar])-1] = 1


smpi = len(insrc)
bsize = 200
epo = 500
numbs = int(smpi / bsize)


def checkmodel(nn,insrc,filesrc,yvar):
    fsi = len(insrc)
    nn.addinval = 1
    nn = ffbp.dofeedforward(nn, insrc, np.zeros((fsi, yvar)))
    nn.addinval = 0
    vcnum = 0
    for ii in range(0, fsi):
        if (np.argmax(nn.odata[nn.arrln - 1][ii]) + 1) == filesrc[ii, len(insrc[0])]:
            vcnum += 1
    print('Now------------------------Verify correct=' '%f' %vcnum)
    return nn

loarr = np.zeros((epo*numbs, 1))
epoch = 0
stime = time.time()
nn.outfc='softmax'



if __name__ == '__main__':
    for i in range(0, epo):
        nrp = np.random.permutation(smpi)
        for j in range(0, numbs):
            insrc_bs = insrc[nrp[j * bsize: (j + 1) * bsize],]
            tgsrc_bs = tgsrc[nrp[j * bsize: (j + 1) * bsize],]

            nn = ffbp.dofeedforward(nn, insrc_bs, tgsrc_bs)
            nn = ffbp.dobp(nn)
            nn = ffbp.doupnnwg(nn)
            loarr[epoch] = nn.lo
            epoch += 1

        eq = np.sum(np.array(nn.er) ** 2)
        print('%d' % i + ',State change:Error=' '%f' % eq + ',Learning rate=' + '%f' % nn.lrate)

    nn = checkmodel(nn, insrc, filesrc, yvar)
    print('DeepLearn Trainning ..............End!')
