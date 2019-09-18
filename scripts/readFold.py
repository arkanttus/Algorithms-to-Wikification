from numpy import *

def readFold(idFold, typeSet):
    filename = "C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/codes-to-create-school-wiki2014/virtual_folds/fixed/" + typeSet + '_' + str(idFold) + '.txt'

    fileId = open(filename, 'r')
    In = loadtxt(fileId, delimiter=" ", dtype='int')
    fileId.close()

    foldIds = In

    return foldIds.reshape(-1,1)
#end