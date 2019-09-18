from numpy import *

#implementação do randSample do MatLab
def randSample(arr, perms):
    v = random.permutation(arr)
    r = v[0:perms]
    return r
#end


def read20perFold(idFold, typeSet, perc):

    filename = "C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/codes-to-create-school-wiki2014/virtual_folds/fixed/" + typeSet + '_' + str(idFold) + '.txt'
    fileId = open(filename, 'r')
    In = loadtxt(fileId, delimiter=" ", dtype='int')
    fileId.close()

    k = round(In.shape[0]*perc)
    docsIds = randSample(In, k)

    return docsIds.reshape(-1,1)
#end