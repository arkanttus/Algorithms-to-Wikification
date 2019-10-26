from numpy import *

def readFold(idFold, typeSet):
    filename = "../colecao/codes-to-create-school-wiki2014/virtual_folds/fixed/" + typeSet + '_' + str(idFold) + '.txt'

    fileId = open(filename, 'r')
    In = loadtxt(fileId, delimiter=" ", dtype='int')
    fileId.close()

    foldIds = In

    return foldIds.reshape(-1,1)
#end
