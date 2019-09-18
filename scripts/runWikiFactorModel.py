from numpy import *
from fSInfoPSGDOresidual import fSInfoPSGDOresidual
from sub2ind import sub2ind
from evalModelByDoc3 import evalModelByDoc3
from read20perFold import read20perFold
from readFold import readFold
from randn2 import *

arr = load("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/5132_loose_wss-wiki2014_vfinal.npz")
#print(arr.files)
dbin = arr['Dbin']
Z = arr['Z'].astype('int')
X = arr['X']
F = arr['F']
titles = arr['titles']
rates = arr['rates']

'''titles = array([["anarchism"],
                ["autism"],
                ["albedo"]])

Z = array([[1,1,1,2,2,2,3,3,3],
           [1,2,3,1,2,3,1,2,3],
           [1,1,0,1,0,1,1,0,0]])
X = array([
          #(:,:,1)
          [[20.4954 ,-0.2642 ,-0.2642],[11.6616 ,-0.1228 ,-0.1228],[33.1403 ,-0.1116 ,-0.1116], [20.2982 ,-0.1712 ,-0.1712 ],
          [6.3981 ,-0.1156 ,-0.1156] ,[8.4533 ,-0.1143 ,-0.1143] ,[-0.1583 ,-0.1600 ,-0.1600] ,[7.3227 ,-0.1830 ,-0.1830 ],
          [13.0691 ,-0.1127 ,-0.1127] ,[13.2318 ,-0.1380 ,-0.1380] ,[12.7851 ,-0.1399 ,-0.1399] ,[4.9442 ,-0.2023 ,-0.2023 ]],

          #(:,:,2)
          [[-0.2642 ,20.4954 ,-0.2642] ,[-0.1228 ,9.6021 ,-0.1228] ,[-0.1116 ,33.8967 ,-0.1116] ,[-0.1712 ,18.3188 ,-0.1712 ],
          [-0.1156 ,3.6140 ,-0.1156]  ,[-0.1143 ,3.5086 ,-0.1143] ,[-0.1600 ,-0.1514 ,-0.1600] ,[-0.1830 ,7.2894 ,-0.1830 ],
          [-0.1127 ,13.0015 ,-0.1127] ,[-0.1380 ,5.8376 ,-0.1380] ,[-0.1399 ,5.6369 ,-0.1399]  ,[-0.2023 ,4.9442 ,-0.2023 ]],

          #(:,:,3)
          [[-0.2642 ,-0.2642 ,20.4954] ,[-0.1228 ,-0.1228 ,17.0146] ,[-0.1116 ,-0.1116 ,41.7694] ,[-0.1712 ,-0.1712 ,15.6561 ],
          [-0.1156 ,-0.1156 ,18.2911] ,[-0.1143 ,-0.1143 ,15.7543] ,[-0.1600 ,-0.1600 ,-0.0781] ,[-0.1830 ,-0.1830 ,7.1143 ],
          [-0.1127 ,-0.1127 ,12.5982] ,[-0.1380 ,-0.1380 ,18.0412] ,[-0.1399 ,-0.1399 ,17.4345] ,[-0.2023 ,-0.2023 ,4.9442 ]]], dtype='double')

F = array([
          [-0.4089 ,-0.4097 ,-0.4092],
          [-0.9994 ,-0.9998 ,-0.9999],
          [-6.9948 ,-7.0504 ,-7.1059]])'''
T = Z[:,Z[0,:] != Z[1,:]]
Z = T

savepredictions = 1

#loss function
loss = 'logistic'
#link function
link = 'sigmoid'

symetric = 0 #a rede é simetrica? Coleção Wikipedia é assimetrica

k = 5 #numero de caracteristicas latentes
EPOCHS = 10 # numero de passos do SGD
epochFrac = 0.1 #fração de +'ve e -'ve pares para usar em cada volta
batchSize = 1 #numero de exemplos em cada atualização

#aprendendo
eta = {}
eta['etaLatent'] = 0.1000#0.000001#0.1000  # 1e-1 = 1x10^-1 = 0.1000 (em matlab) / aprendendo porcentagem para caracteristica latente
eta['etaRowBias'] = 0.1000#0.000001#0.1000 #aprendendo estatistica para no Bias
eta['etaLatentScaler'] = 0.1000#0.000001#0.1000 #aprendendo estatistica para escalar de caracteristicas latentes
eta['etaBias'] = 0.1000#0.000001#0.1000

#penalização de termos
varLambda = {}
varLambda['lambdaLatent'] = 0.0100#0.000100  #1e-2 = 1x10^-2 = 0.0100 em matlab  /  regularization for node's latent vector U
varLambda['lambdaRowBias'] = 0.0100#0.000100#0.0100 #regularization for node's bias UBias
varLambda['lambdaLatentScaler'] = 0.0100#0.000100#0.0100 #regularization for scaling factors Lambda (in paper)

varLambda['lambdaPair'] = 0.00001#0.00100#0.00001 #1e-5 = 1x10^-5 = 0.00001 em matlab / regularization for weights on pair features
varLambda['lambdaBilinear'] = 0.00001#0.00100#0.00001 #regularization for weights on node features

varLambda['lambdaScaler'] = 1  #scaling factor for regularization, can be set to 1 by default

F = F[:, 0:5132]
titles = titles[0:5132, :]

m = titles.shape[0] #numero de nos

sideBilinear = F #caracteristicas do artigo
sidePair = X  #caracteristicas de link (pair)

dBilinear = sideBilinear.shape[0] #numero de caracteristicas de artigos
nodeFeaturesPresent = double(dBilinear > 0)

dPair = sidePair.shape[0]  #numero de caracteristicas de link (pair)
linkFeaturesPresent = double(dPair > 0)

eta['etaPair'] = linkFeaturesPresent * 0.001 #1e-3 /  learning rate for pairs, when their features are present
eta['etaBilinear'] = nodeFeaturesPresent * 0.001 #learning rate for node, when their features are present
eta['etaBias'] = linkFeaturesPresent * 0.001 #learning rate for global bias, used when features are present

trainFrac = 1 #training set fraction for learning

# This script aims to evaluate the factor model to predict link through 5-fold cross validation methodology

ats_auc, ats_f1, ats_rmse, ats_prec, ats_rec = 0,0,0,0,0

maxFolds = 1

for i in range(0,maxFolds):
    random.seed(1) # to ensure reproducible experiments
    #print(random.get_state())

    #initialize weight (model) parameters

    weights = {}
    weights['U'] =  1/sqrt(k) * randn2(k,m)  #for k latent features and m nodes
    weights['P'] =  1/sqrt(k) * randn2(k,m)  #for k latent features and m nodes
    weights['Q'] =  1/sqrt(k) * randn2(k,m)  #for k latent features and m nodes

    weights['UBias'] = randn2(m,1)

    weights['ULatentScaler'] = diag(randn2(k,1).ravel()) #for asymmetric use randn(k, k); for symmetric use diag(randn(k, 1))
    weights['GLatentScaler'] = diag(randn2(k,1).ravel())
    weights['ULatentScaler'] = diag(randn2(k,1).ravel()) #for asymmetric use randn(k, k); for symmetric use diag(randn(k, 1))

    weights['WPair'] = linkFeaturesPresent * randn2(1,dPair) # for dPair features for each pair
    weights['WBias'] = linkFeaturesPresent * randn2()
    weights['WBilinear'] =  nodeFeaturesPresent * randn2(dBilinear, dBilinear) # V

    Dtr = array([])
    idsTr = read20perFold(i+1, 'train', trainFrac)
    #idsTr = readFold(i+1, 'train')
    #k = 1

    #for i in idsTr:
     #   print("{}: {}".format(k, i))
      #  k += 1

    ITr = in1d(Z[0,:],idsTr).ravel().nonzero()[0]

    #print(ITr)
    #print(ITr[0])
    #print(Z)
    #print(in1d(Z[0,:],idsTr))

    Dtr = Z[:,ITr]

    #print(Dtr[0])
    #print(Dtr.shape[2])
   # print(Dtr)

    insU = unique(Dtr[0,:]) #count # of nodes in training set

    #print(insU.shape[0])

    #obtain # of pairs by articles in training set
    npairsTr = zeros((insU.shape[0],1), dtype='int')
    for j in range(0, insU.shape[0]):
        npairsTr[j] = size( Dtr[:, Dtr[0,:]==insU[j]], axis=1 )

   # print("npairsTr: {}".format(npairsTr))

    convergenceScoreTr = {}
    convergenceScoreTr['D'] = Dtr
    convergenceScoreTr['npairs'] = npairsTr

    idsTs = readFold(i+1, 'test')
    ITs = in1d( Z[0,:], idsTs ).ravel().nonzero()[0]
    Dts = Z[:, ITs]

    insU = unique(Dts[0,:])

   # print("Dts: {}".format(Dts))
    #print("insU shape 0:  {}".format(insU.shape[0]))

    npairsTs = zeros( (insU.shape[0],1))
    for j in range(0, insU.shape[0]):
        npairsTs[j] = size( Dts[:, Dts[0,:]==insU[j]], axis=1 )

    convergenceScore = {}
    convergenceScore['D'] = Dts
    convergenceScore['npairs'] = npairsTs

    #training the factorization algorithm
    #the factorization algorithm is trained on train set (Dtr) in order to create a model of link prediction
    #the model is characterized by the set of output weights such as U, P,
    #Wpair, etc ...

    U, P, Q, UBias, ULatentScaler, GLatentScaler, WPair, WBias, WBilinear = fSInfoPSGDOresidual(Dtr, sidePair, sideBilinear, weights, varLambda, eta, EPOCHS, epochFrac, convergenceScoreTr, convergenceScore, loss, link, symetric, [[]] )

    #test the factorization model
    #the factorization model is used to perform predictions on test set
    #(Dts), i.e. data that still has been unseen before by the algorithm
    #(not train)

    SPred = (U @ ULatentScaler @ U.conj().transpose()) + (UBias.conj().transpose() + UBias)  # bsxfun(@plus) em matlab
    SPred = SPred + (P @ GLatentScaler @ Q.conj().transpose())

    if sideBilinear.size > 0:
        SPred = SPred + sideBilinear.conj().transpose() @ WBilinear @ sideBilinear + WBias #node features present

    if sidePair.size > 0:
        SPred = SPred + reshape(WPair @ reshape(sidePair, concatenate((array([dPair]), array([m * m])))),concatenate((array([m]), array([m]))))

    PPred = 1/(1+exp(-SPred)) # predicted probability between 0 and 1

    #evaluate test set performance
    #I select only the entries in PPred (i.e., predictions) which
    #representing pairs of articles on test set
    testLinks = sub2ind(PPred.shape, Dts[0,:], Dts[1,:])
    testLinks = testLinks - 1 #como testLink servirá de índice, subtraímos 1, ja que os indices dos arrays começam em 0
    predictions = PPred.flatten('F')[testLinks] #transforma a matriz em um vetor linha, começando pelas colunas (Ordem Fortran, igual ao matlab)
    predictions = reshape(predictions, (1, predictions.shape[0]))
    #real = Dts[2,:]
    real = Dts[2,:].reshape(1,Dts.shape[1])

    npairsTs = convergenceScore['npairs']

    ts_auc, ts_rmse, ts_f1, ts_prec, ts_rec = evalModelByDoc3(predictions, real,npairsTs)

    ats_auc = ats_auc + ts_auc
    ats_rmse = ats_rmse + ts_rmse
    ats_f1 = ats_f1 + ts_f1
    ats_prec = ats_prec + ts_prec
    ats_rec = ats_rec + ts_rec

    #each train/test evaluation, call as holdout, is print to obtain
    #partial performance

    print('holdout {}: avg: auc = {} f1 = {:.4f} prec = {:.2f} rec = {} rmse = {:.5f}\n'.format(int(i+1), double(ts_auc), double(ts_f1), double(ts_prec), double(ts_rec), double(ts_rmse)))
    #print('holdout {}: avg: auc = {} f1 = {} prec = {} rec = {} rmse = {}\n'.format(str(int(i+1)), str(ts_auc), str(ts_f1), str(ts_prec), str(ts_rec), str(ts_rmse)))

#the average performance of algorithm after execution of n-fold cross-validation
print('avg: auc: {} f1: {:.4f} rmse: {:.5f} \n '.format( double(ats_auc/maxFolds), double(ats_f1/maxFolds), double(ats_rmse/maxFolds)))
#print('avg: auc: {} f1: {} rmse: {} \n '.format(str(ats_auc/maxFolds), str(ats_f1/maxFolds), str(ats_rmse/maxFolds)))