from numpy import *
from evalModel import *

def evalModelByDoc3(predictions, truth, vp):

    #TESTE
    #seterr(all='print')

    #Summary of this function goes here
    #  This function computes the error predictions regarding
    #  popular metrics widely used to evaluate machine learning algorithm
    #  such as AUC (Area Under the Curve), RMSE (Root Mean Square Error),
    #  Precision and Recall

    vp = vp.astype('int')

    #  threshold default is 0.5
    optarg = array([[0.5]])
    epsilon = optarg[0][0]

    offset = -1
    n = vp.shape[0]

    mauc = array([])
    mf1 = array([])
    mprec = array([])
    mrec = array([])

    for i in range(0,n):
        p = predictions[0, offset+1 : offset + int(vp[i]) ]
        p = p.reshape((1, p.shape[0]))
        t = truth[0, offset+1 : offset + int(vp[i]) ]
        t = t.reshape((1,t.shape[0]))

        auc, rmse = evalModel(p, t)
        #mauc = concatenate((mauc, auc))
        mauc = block([mauc,auc])

        #print(mauc)

        bPs = (p > epsilon)

        fp = sum( (bPs == 1) & (t == 0) )
        fn = sum( (bPs == 0) & (t == 1) )
        tp = sum( (bPs == 1) & (t == 1) )

        prec = array([tp / (tp + fp)])
        rec = array([tp / (tp + fn)])

        mprec = concatenate((mprec, prec))
        mrec = concatenate((mrec, rec))

        offset = offset + vp[i]
        offset = offset[0]

    #END FOR

    #print(mauc)
    auc = nanmean(mauc) #equivalente ao nanmean(mauc) em matlab
    prec = nanmean(mprec, axis=0)
    rec = nanmean(mrec, axis=0)

    f1 = 2 * outer(prec,rec) / (prec + rec)

    if isnan(f1):
        f1 = 0

    return (auc, rmse, f1, prec, rec)