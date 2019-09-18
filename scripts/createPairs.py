from numpy import *

def createPairs(D,K):

    # transform the squared matrix D of m-d to 3 x pairs matrix Dbin
    c0,r0 = where(D.transpose()==0)
    c1,r1 = where(D.transpose()==1)

    temp = block([[r0,r1],[c0,c1],[ zeros((1,r0.shape[0])), ones((1,r1.shape[0])) ]])
    temp = temp.transpose()
    temp = temp[temp[:,0].argsort()] #sortrows matlab
    Dbin = temp.transpose()

    # define a 3 x pairs matrix of pairs which  necessarily contains a label
    c0, r0 = where(K.transpose() == 0)
    c1, r1 = where(K.transpose() == 1)

    temp = block([[r0, r1], [c0, c1], [zeros((1, r0.shape[0])), ones((1, r1.shape[0]))]])
    temp = temp.transpose()
    temp = temp[temp[:, 0].argsort()]  # sortrows matlab
    temp = temp.transpose()

    Z = Dbin[:,nonzero(temp[2,:])]
    Z = Z.reshape((Z.shape[0] , Z.shape[2]))

    return (Dbin, Z)