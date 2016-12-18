from __future__ import division
import numpy as np
import formula as fa
import neuralnet
import os

def dofeedforward(nn, insrc_bs, tgsrc_bs):
    si = len(insrc_bs)

    nn.odata[0] = np.concatenate((np.ones((si, 1)), insrc_bs), axis=1)

    for i in range(1, nn.arrln-1):
        if nn.actfc is 'sigm':
            nn.odata[i] = fa.sigm(np.dot(nn.odata[i-1], nn.wg[i - 1].T))
        if nn.actfc is 'tanh_opt':
            nn.odata[i] = fa.tanh_opt(np.dot(nn.odata[i - 1], nn.wg[i - 1].T))

        if nn.dpf > 0:
            if nn.addinval:
                nn.odata[i] = nn.odata[i] * (1 - nn.dpf)
            else:
                nn.dom[i] = np.greater(np.random.rand(len(nn.odata[i]), len(nn.odata[i][0])), nn.dpf) * 1
                nn.odata[i] = nn.odata[i] * nn.dom[i]
        nn.odata[i] = np.concatenate((np.ones((si, 1)), nn.odata[i]), axis=1)
    if nn.outfc is 'sigm':
        nn.odata[nn.arrln - 1] = fa.sigm(np.dot(nn.odata[nn.arrln - 2], nn.wg[nn.arrln - 2].T))
    if nn.outfc is 'linear':
        nn.odata[nn.arrln - 1] = np.dot(nn.odata[nn.arrln - 2], nn.wg[nn.arrln - 2].T)
    if nn.outfc is 'softmax':
        nn.odata[nn.arrln - 1] = np.dot(nn.odata[nn.arrln - 2], nn.wg[nn.arrln - 2].T)
        nn.odata[nn.arrln - 1] = np.exp(fa.bsxfun('minus', nn.odata[nn.arrln - 1], np.matrix(nn.odata[nn.arrln - 1]).max(1)))
        nn.odata[nn.arrln - 1] = fa.bsxfun('rdivide', nn.odata[nn.arrln - 1], np.matrix(nn.odata[nn.arrln - 1]).sum(1))
    nn.er = tgsrc_bs - nn.odata[nn.arrln - 1]

    if nn.outfc is 'softmax':
        nn.lo = np.sum(np.array(tgsrc_bs) * np.log(nn.odata[nn.arrln - 1])) / si
    else:
        nn.lo = 1/2 * np.sum(nn.er**2) / si

    return nn

def dobp(nn):
    sparsityError = 0

    bpsm = [[]] * nn.arrln
    if nn.outfc is 'sigm':
        bpsm[nn.arrln - 1] = -nn.er * (nn.odata[nn.arrln - 1] * (1 - nn.odata[nn.arrln - 1]))
    else:
        bpsm[nn.arrln - 1] = -nn.er
    for i in range(nn.arrln - 1, 1, -1):
        if nn.actfc is 'sigm':
            d_act = nn.odata[i-1] * (1 - nn.odata[i-1])
        if nn.actfc is 'tanh_opt':
            d_act = 1.7159 * 2/3 * (1 - 1/1.7159 ** 2 * nn.odata[i-1] ** 2)
        if i + 1 == nn.arrln :
            bpsm[i-1] = ( np.dot(bpsm[i], nn.wg[i-1]) + sparsityError) * d_act
        else:
            bpsm[i-1] = (np.dot(bpsm[i][0:len(bpsm[i]), 1:len(bpsm[i][0])], nn.wg[i-1]) + sparsityError) * d_act

        if nn.dpf > 0:
            bpsm[i-1] = bpsm[i-1] * np.concatenate(np.ones((len(bpsm[i-1]), 1)), nn.dom[i])
    for i in range(0, nn.arrln-1):
        if i+1 == nn.arrln-1:
            nn.dwg[i] = np.dot(bpsm[i+1].T, nn.odata[i]) / len(bpsm[i+1])
        else:
            tempbpsm = bpsm[i+1]
            nn.dwg[i] = np.dot(bpsm[i+1][0:len(tempbpsm), 1:len(tempbpsm[0])].T, nn.odata[i]) / len(tempbpsm)
    return nn

def doupnnwg(nn):
    for i in range(0, nn.arrln - 1):
        fix_dwg = nn.dwg[i]
        fix_dwg = nn.lrate * fix_dwg

        if nn.momentum > 0:
            nn.wgv[i] = nn.wgv[i]*nn.momentum + fix_dwg
            fix_dwg = nn.wgv[i]
        nn.wg[i] = nn.wg[i] - fix_dwg
    return nn
