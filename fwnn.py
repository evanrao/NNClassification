from __future__ import division
import math,copy,os,matstorage
import numpy as np
from neural import Neural

global global_wf,theHi,wf_Hi_Xi_Ui

insrc = matstorage.dataload('in_data.npy')
tgsrc = matstorage.dataload('tg_data.npy')

nn = Neural(len(insrc),len(insrc[0]),2,1)
global_wf = [[1] * nn.Hi] * 1 # or use np.ones((1, nn.Hi))
theHi=0
wf_Hi_Xi_Ui= [[0] * nn.Xi] * nn.Hi

c1=np.random.rand(nn.Xi,nn.Ui)
sigma1=0.2 * np.ones((nn.Xi,nn.Ui))
wf_wnn=np.random.rand(nn.Xi,nn.Hi)
c1_wnn=np.ones((nn.Xi,nn.Hi))
sigma1_wnn=np.ones((nn.Xi,nn.Hi))

innor = matstorage.log_normalization(insrc)
tgnor = matstorage.log_normalization(tgsrc)

for i in range(0, nn.Xi):
    maxXi=np.max(innor[0:nn.Si,i])
    minXi=np.min(innor[0:nn.Si,i])
    c1_wnn[i,0:nn.Hi]=c1_wnn[i,0:nn.Hi] * 0.5 * (maxXi+minXi)
    sigma1_wnn[i,0:nn.Hi]=sigma1_wnn[i,0:nn.Hi] * 0.2 * (maxXi-minXi)

eta=0.1
epo=1000 #The end of the cycle, the maximum number of training
epoch=1 #The starting point of the loop
error=0.1 #Initialization error square sum
err=0.0001 #The sum of the square of the expected errors
errorpn=1 #Error counter for individual samples


Err_NetOut2Show=[]  #Error history data set

def anfiswfsum(theXi,outLayer1,beforeValue,beforeMat):
    """
    Wf the initial value of 1, the global variable is set to global_wf [Hi],
    the xi is the current xi subscript incoming, initialized to 0.
    """
    global theHi,global_wf,wf_Hi_Xi_Ui
    for u in range(0, nn.Ui):
        if (theHi <= nn.Hi-1):
            afterValue = beforeValue * outLayer1[theXi, u]
            afterMat = beforeMat
            afterMat[0,theXi] = u
            if (theXi == nn.Xi-1):
                global_wf[0,theHi] = afterValue
                wf_Hi_Xi_Ui[theHi,0:len(wf_Hi_Xi_Ui[0])]=afterMat
                theHi = theHi + 1
        if (theHi <= nn.Hi-1):
            if theXi < nn.Xi-1:
                anfiswfsum(theXi +1, outLayer1, afterValue, afterMat)

if __name__ == '__main__':
    print ('FWNN Training Start...........................')
    # Start the training sample
    while (error>err and epoch<=epo):
        error = 0
        for si in range(0, nn.Si):
            #Reinitialize the data
            outLayer1 = np.zeros((nn.Xi, nn.Ui))
            outLayer4 = np.zeros((1, nn.Hi))
            wf = np.zeros((1, nn.Hi))
            wfDraw = np.zeros((1, nn.Hi))
            outWnn = np.zeros((1, nn.Hi))
            outWnnXi = np.ones((nn.Xi, nn.Hi))

            global_wf = np.ones((1, nn.Hi))
            wf_Hi_Xi_Ui = np.zeros((nn.Hi, nn.Xi))
            theHi = 1

            deLayer1 = np.zeros((nn.Xi, nn.Ui))
            deltac1 = np.zeros((nn.Xi, nn.Ui))
            deltasigma1 = np.zeros((nn.Xi, nn.Ui))

            deLayer3 = np.zeros((1, nn.Hi))
            deLayer3Wnn = np.zeros((1, nn.Hi))
            delwf_wnn = np.random.rand(nn.Xi, nn.Hi)
            delc1_wnn = np.ones((nn.Xi, nn.Hi))
            delsigma1_wnn = np.ones((nn.Xi, nn.Hi))

            outLayer5 = np.zeros((nn.Si, 1))
            # Compute layer 1
            for i in range(0, nn.Xi):
                for u in range(0, nn.Ui):
                    # Set the membership function
                    outLayer1[i, u] = math.exp(-0.5 * ((innor[si, i] - c1[i, u]) ** 2 / sigma1[i, u]) ** 2)
            # Compute layer 2, using recursive computation  W(u)=Ui(X1)*Ui(X2)*Ui(X3)*Ui(X4)
            if (nn.Hi == nn.Ui):
                for j in range(0, nn.Hi):
                    wf[j] = 1
                    for i in range(0, nn.Xi):
                        wf[j] = wf[j] * outLayer1[i, j]
                        wf_Hi_Xi_Ui[j, i] = j
            else:
                anfiswfsum(0, outLayer1, 1, np.zeros((1,nn.Xi)))
                wf = copy.deepcopy(global_wf)
            # Compute layer 3 - the third layer of F
            if np.sum(wf[0,0:len(wf[0])]) != 0:
                for j in range(0, nn.Hi):
                    wfDraw[0, j] = wf[0, j] / np.sum(wf[0,0:len(wf[0])])
            else:
                print ('sum(wf(0,:))=0 Of the error, exit the system!')
                os._exit()
            # Compute layer 3 - the third layer of W
            for j in range(0, nn.Hi):
                for i in range(0, nn.Xi):
                    tempwnnz = ((innor[si, i] - c1_wnn[i, j]) / sigma1_wnn[i, j]) ** 2
                    outWnnXi[i, j] = wf_wnn[i, j] * (abs(sigma1_wnn[i, j]) ** (-1 / 2)) * (1 - tempwnnz) * math.exp(-0.5 * tempwnnz)
                outWnn[0,j] = np.sum(outWnnXi[0:len(outWnnXi), j])
            # Compute layer 4
            for j in range(0, nn.Hi):
                outLayer4[0, j] = wfDraw[0, j] * outWnn[0,j]
            # Compute layer 5

            outLayer5[si, 0] = np.sum(outLayer4[0,0:len(outLayer4[0])])
            errorpn = (outLayer5[si, 0] - tgnor[si,0])
            error += (abs(tgnor[si,0] - outLayer5[si, 0])) ** 2
            # Directly start the least squares method of multiple linear regression, calculate the optimal value of P, only once
            # Gradient algorithm, the reverse calculation, layer 4 of the incremental adjustment
            for j in range(0, nn.Hi):
                deLayer3Wnn[0, j] = errorpn * wfDraw[0, j]
                deLayer3[0,j] = errorpn * outWnn[0,j]

            for j in range(0, nn.Hi):
                for i in range(0, nn.Xi):
                    deltempwnnz = (innor[si, i] - c1_wnn[i, j]) / sigma1_wnn[i, j]
                    delwf_wnn[i, j] = deLayer3Wnn[0, j] * (abs(sigma1_wnn[i, j]) ** (-1 / 2)) * (1 - deltempwnnz ** 2) * math.exp(-0.5 * deltempwnnz ** 2)

                    delc1_wnn[i, j] = deLayer3Wnn[0, j] * wf_wnn[i, j] * (abs(sigma1_wnn[i, j]) ** (-3 / 2)) * math.exp(-0.5 * deltempwnnz ** 2) * ((3 * deltempwnnz - deltempwnnz ** 3))
                    delsigma1_wnn[i, j] = deLayer3Wnn[0, j] * wf_wnn[i, j] * (abs(sigma1_wnn[i, j]) ** (-3 / 2)) * math.exp(-0.5 * deltempwnnz ** 2) * (3.5 * deltempwnnz ** 2 - deltempwnnz ** 4 - 0.5)
            # Modify the weight value
            wf_wnn = wf_wnn - eta * delwf_wnn
            c1_wnn = c1_wnn - eta * delc1_wnn
            sigma1_wnn = sigma1_wnn - eta * delsigma1_wnn

        if error < 0.5:
            eta = 0.05
        Err_NetOut2Show.append(error) # Save each error
        print('%d' %epoch + ',State change:Error=' '%f' %error + ',Learning rate=' + '%f' %eta )

        epoch += 1
    """To be continued ...
    classification calculation function
    """
    if epoch < epo:     #The fit ends and the model parameters are saved if the error reaches the expected value
        print 'The fit is as expected and the model parameters are saved...'
        np.save('./mdata/rherror.npy',error)
        np.save('./mdata/rhc1.npy', c1)
        np.save('./mdata/rhsigma1.npy', sigma1)
        np.save('./mdata/rhwf_wnn.npy', wf_wnn)
        np.save('./mdata/rhc1_wnn.npy', c1_wnn)
        np.save('./mdata/rhsigma1_wnn.npy', sigma1_wnn)
