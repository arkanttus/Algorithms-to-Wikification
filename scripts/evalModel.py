from numpy import *
from scipy.stats import rankdata

def evalModel (predictions, truth):

    # TESTE
    #seterr(all='ignore')

    n = predictions.shape[1] #num de colunas
    rmse = sqrt( sum( (predictions - truth) ** 2 ) / n )

    # Count observations by class
    nTarget     = sum( double(truth == 1) )
    nBackground = sum( double(truth != 1) )

    #rank data
    R = rankdata(predictions) #tiedrank in matlab

    R = R.reshape((1,R.shape[0]))
    R = R.astype('int')

    #calculate AUC
    a1 = sum(R[(truth==1)])
    a2 = (nTarget ** 2 + nTarget)/2
    a3 = nTarget * nBackground
    auc = round(( round(sum(R[(truth==1)]),8) - round((nTarget ** 2 + nTarget)/2, 8) ) / round((nTarget * nBackground),8), 8)

    #print(a1)
    #print(a2)
    #print(a3)
    #print(auc)

    return (auc, rmse)