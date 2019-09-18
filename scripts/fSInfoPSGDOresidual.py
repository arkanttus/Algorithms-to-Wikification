from numpy import *
from evalModelByDoc3 import *
from sub2ind import *

def fSInfoPSGDOresidual (D, sidePair, sideBilinear, weights, varLambda, eta, EPOCHS, epochFrac, convergenceScoreTr, convergenceScore, loss,
                         link, symetric, covariance):

    pairs = size(D, axis=1);
    print("Processando {} pairs le{}".format(pairs, round( log10(pairs) )) )

    varSquare = (loss == "square")
    squareHinge = (loss == "squareHinge")
    sigmoid = (link == "sigmoid")

    # Extracting weights from the appropriate structures
    UO, PO, QO = weights["U"], weights["P"], weights["Q"]
    UOBias, ULatentScaler = weights["UBias"], weights["ULatentScaler"]
    GLatentScaler = weights["GLatentScaler"]
    WPair, WBias, WBilinear = weights["WPair"], weights["WBias"], weights["WBilinear"]

    #utilizando dicionarios do python para substituir as structs do matlab
    if type(varLambda) != dict:     # isstruct(lambda)
        lambdaLatent, lambdaRowBias, lambdaLatentScaler, lambdaPair, lambdaBilinear, lambdaScaler = varLambda, varLambda, varLambda, varLambda, varLambda, ones((UOBias.shape))
    else:
        lambdaLatent, lambdaRowBias, lambdaLatentScaler, lambdaPair, lambdaBilinear, lambdaScaler = varLambda["lambdaLatent"], varLambda["lambdaRowBias"], varLambda["lambdaLatentScaler"], varLambda["lambdaPair"], varLambda["lambdaBilinear"], varLambda["lambdaScaler"]

    if len(str(lambdaScaler)) == 1:
        lambdaScaler = dot(lambdaScaler, ones((UOBias.shape)))

    if type(eta) != dict:
        linkFeaturesPresent, nodeFeaturesPresent = 1 , 1
        etaLatent0, etaLatentScaler0, etaPair0, etaBilinear0, etaRowBias0, etaBias0 = eta, eta, eta*linkFeaturesPresent, eta*nodeFeaturesPresent, eta, eta*nodeFeaturesPresent
    else:
        etaLatent0, etaLatentScaler0, etaPair0, etaBilinear0, etaRowBias0, etaBias0 = eta["etaLatent"], eta["etaLatentScaler"], eta["etaPair"], eta["etaBilinear"], eta["etaRowBias"], eta["etaBias"]

    lowRank = ((WBilinear.shape[0]) != (WBilinear.shape[1])) #verifica se o numero de linhas é diferente do numero de colunas
    cachedCovariance = len(covariance[-1]) > 0  #utilizando array com dtype=object para substituir a cell do matlab
    hasDyadicSideInfo = sidePair.size > 0

    U, P, Q, UBias, = UO, PO, QO, UOBias
    UOld, POld, QOld, UBiasOld = UO, PO, QO, UOBias
    ULatentScalerOld, GLatentScalerOld = ULatentScaler, GLatentScaler
    WPairOld, WBiasOld, WBilinearOld = WPair, WBias, WBilinear

    lastScore, bestScore, badEpochs = 0,0,0
    trainError, testError = empty((0)), empty((0)) # ou [] (lista vazia)

    for e in range(0,EPOCHS):
        etaLatent = etaLatent0 / ( (1 + etaLatent0 * lambdaLatent) * (e+1) )
        etaRowBias = etaRowBias0 /( (1 + etaRowBias0 * lambdaRowBias) * (e+1) )
        etaLatentScaler = etaLatent0 / ((1 + etaLatentScaler0 * lambdaLatentScaler) * (e+1) )
        etaPair = etaPair0 / ( (1 + etaPair0 * lambdaPair) * (e+1) )
        etaBilinear = etaBilinear0 / ( (1 + etaBilinear0 * lambdaBilinear) * (e+1) )
        etaBias = etaBias0 / ( (1 + etaBias0 * lambdaLatent) * (e+1) )
        lossV = 0

        I = random.permutation(range(0,pairs)) #randperm em matlab, retorna um array com elementor de 0 a pairs-1, em ordem aleatoria
        D = D[:,I]

        for t in range(0,round( epochFrac * pairs )+1): #percorre epochFrac por cento do conjunto
            examples = t #pares

            i = int(D[0,examples]-1)
            j = int(D[1,examples]-1)
            truth = int(D[2,examples])

            #predição     E x 1

           # print(U.conj().transpose())
            #U = U.conj().transpose()
           # print(U)
            #print(ULatentScaler)

            prediction = ( U[:,i].reshape(1,U.shape[0]) @ ULatentScaler @ U[:,j].reshape(U.shape[0],1) + P[:,i].reshape(1,P.shape[0]) @ GLatentScaler @ Q[:,j].reshape(Q.shape[0],1) + UBias[i] + UBias[j] )[0][0]

           # prediction = ( U[:,i].conj().transpose() @ ULatentScaler @ U[:,j] + P[:,i] @ GLatentScaler @ Q[:j] + UBias[i] + UBias[j]).conj().transpose()
            #print(prediction)

            if hasDyadicSideInfo:
                prediction = prediction + WPair @ sidePair[:,i,j] + WBias

            if etaBilinear > 0:
                if lowRank:
                    prediction = prediction +  ( WBilinear @ sideBilinear[:,i]).conj().transpose() @ ( WBilinear @ sideBilinear[:,j] )
                else:
                    prediction = prediction + sideBilinear[:,i].conj().transpose() @ WBilinear @ sideBilinear[:,j]

            #Link function
            if sigmoid:
                prediction = 1/( 1 + exp(-prediction) )

            #Gradients
            #gradients were implemented (in pseudocode) between the lines 10 and 18

            #Common gradient Scaler
            gradScaler = (prediction - truth)[0] #code linha 10

            if varSquare:
                gradScaler = 2 * gradScaler
                if sigmoid:
                    gradScaler = gradScaler * prediction * (1 - prediction)

            gradI = ULatentScaler @ U[:,j].reshape(U.shape[0],1) #linha 11
            gradJ = ULatentScaler.conj().transpose() @ U[:,i].reshape(U.shape[0],1) #linha 12
            gradP = GLatentScaler @ Q[:,j].reshape(Q.shape[0],1) #linha 13
            gradQ = GLatentScaler.conj().transpose() @ P[:,i].reshape(P.shape[0],1) #linha 14
            #gradRowBias = ones((1,len(str(examples))), dtype='int')
            gradRowBias = ones((1, 1), dtype='int')
            #gradBias = ones((1,len(str(examples))), dtype='int')
            gradBias = ones((1, 1), dtype='int')

            if hasDyadicSideInfo: #linha 15
                gradPair = sidePair[:,i,j].conj().transpose()

            if etaLatentScaler > 0:
                gradLatentScaler = diag( U[:,i] * U[:,j] ) #linha 16
                gradPQ = diag( P[:,i] * Q[:,j] ) #linha 17

            if etaBilinear > 0:
                if lowRank:
                    #alterei a multiplicação de @ para outer, por ser mult de vetores
                    #talvez tenha que alterar o transpose para reshape
                    gradBilinear = outer(outer(WBilinear, sideBilinear[:,i]), sideBilinear[:,j].conj().transpose()) + outer(outer(WBilinear , sideBilinear[:,j] ) , sideBilinear[:,i].conj().transpose())
                else:
                    if cachedCovariance:
                        gradBilinear = covariance[i][j]
                    else:
                        gradBilinear = outer(sideBilinear[:,i],sideBilinear[:,j].conj().transpose()) #talvez tenha que alerar tranpose para reshape

            # If relationship is symmetric, then update not only for (i,j) but for (j,i) also
            # Wikipedia graph is originally asymetric
            if symetric:
                # Actually, I think only the bilinear component need be
                # updated here, although updating the rest doesn't hurt...
                gradI = gradI + ULatentScaler.conj().transpose() @ U[:,j].reshape(U.shape[0],1)
                gradJ = gradJ + ULatentScaler @ U[:,i].reshape(U.shape[0],1)
                gradRowBias = gradRowBias + gradRowBias
                gradBias = gradBias + gradBias

                if hasDyadicSideInfo:
                    gradPair = gradPair + sidePair[:,j,i].conj().transpose()

                if etaLatentScaler > 0:
                    gradLatentScaler = gradLatentScaler + gradLatentScaler.conj().transpose()
                # hasNodeSideInfo n existe
                if hasDyadicSideInfo and etaBilinear > 0:
                    gradBilinear = gradBilinear + gradBilinear.conj().transpose()

            #  Updates all model weights - lines 19-28

            #print((etaLatent * ( gradScaler * concatenate((gradI, gradJ)) +  lambdaLatent * concatenate(( lambdaScaler[i] * U[:,i].reshape(U.shape[0],1), lambdaScaler[j] * U[:,j].reshape(U.shape[0],1) )))))

            #fuckings linhas que deram uma dorzinha de cabeça
            U[:,[i,j]] = U[:, [i,j]] - etaLatent * ( gradScaler * concatenate((gradI, gradJ), axis=1) +  lambdaLatent * concatenate(( lambdaScaler[i] * U[:,i].reshape(U.shape[0],1), lambdaScaler[j] * U[:,j].reshape(U.shape[0],1) ),axis=1))
            #P[:,i] = P[:,i].reshape(P.shape[0],1) - etaLatent * ( gradScaler * gradP + lambdaLatent * lambdaScaler[i] * P[:,i].reshape(P.shape[0],1) )
            P[:, i] = (P[:, i].reshape(P.shape[0], 1) - etaLatent * (gradScaler * gradP + lambdaLatent * lambdaScaler[i] * P[:, i].reshape(P.shape[0], 1))).ravel()
            #Q[:,j] = Q[:,j] - etaLatent @ ( outer( gradScaler, gradQ) + lambdaLatent @ lambdaScaler[j] @ Q[:,j])
            Q[:,j] = (Q[:,j].reshape(Q.shape[0],1) - etaLatent * (gradScaler * gradQ + lambdaLatent * lambdaScaler[j] * Q[:,j].reshape(Q.shape[0],1))).ravel()

            #talvez tenha que alterar o indice de UBias
            UBias[ range(i,j) ] = UBias[ range(i,j) ] - etaRowBias * (  gradScaler * gradRowBias  + lambdaRowBias * UBias[ range(i,j)] )
            WBias = WBias - etaBias * gradScaler * gradBias

            if hasDyadicSideInfo:
                WPair = WPair - etaPair * (gradScaler * gradPair + lambdaPair * WPair )

            if etaLatentScaler > 0:
                ULatentScaler = ULatentScaler - etaLatentScaler * ( gradScaler * gradLatentScaler  + lambdaLatentScaler * ULatentScaler )
                GLatentScaler = GLatentScaler - etaLatentScaler * ( gradScaler * gradPQ  + lambdaLatentScaler * GLatentScaler )

            if etaBilinear > 0:
                WBilinear = WBilinear - etaBilinear * ( gradScaler * gradBilinear + lambdaBilinear * WBilinear )

        #end for interno

        #Codigo Opcional
        #Checa a informação periodica (for cada 10 epochs) acerca da performance do modelo de predição no conjuto de teste
        #In order to do that, this code compute the objective function to eval how far the actual link status is from prediction status

        if D.size > 0 and mod(e+1,10) == 0:
            print('epoch {} '.format(e+1))
            # trainError = convergenceScoreTr(U',UBias,ULatentScaler,WPair,WBias,WBilinear); newScoreTr = trainError.auc;
            # testError = convergenceScore(U',UBias,ULatentScaler,WPair,WBias,WBilinear); newScore = testError.auc;

            #observe que estou usando a mesma função acima em ordem para avaliar a performance da predição de link no conjunto de teste
            SPred = ( U.conj().transpose() @ ULatentScaler @ U ) + ( UBias.conj().transpose() + UBias ) #bsxfun(@plus) em matlab
            SPred = SPred + ( P.conj().transpose() @ GLatentScaler @ Q )

            if sideBilinear.size > 0:
                SPred = SPred + sideBilinear.conj().transpose() @ WBilinear @ sideBilinear + WBias

            if sidePair.size > 0:
                m = UBias.conj().transpose().shape[1] #numero colunas
                dPairs = sidePair.shape[0] #numero linhas
                SPred = SPred + reshape( WPair @ reshape(sidePair, concatenate(( array([dPairs]), array([m*m])))), concatenate(( array([m]), array([m]))))
                #print(SPred)
                #SPred = SPred + reshape( WPair, reshape(sidePair, concatenate(( array([dPairs]), array([m*m]) ) ), concatenate(( array([m]), array([m])), axis=1) )

            PPred = 1 / (1 + exp(-SPred)) #valor da probabilidade de predição entre 0 e 1
            #print(PPred)

            #Seleciono apenas as entradas em PPred (i.e., predictions) onde
            #representam pares de artigos no conjunto de testes

            testLinks = sub2ind(PPred.shape, convergenceScore["D"][0,:], convergenceScore["D"][1,:] )
            testLinks = testLinks - 1 #como testLink servirá de índice, subtraímos 1, ja que os indices dos arrays começam em 0
            predictions = PPred.flatten('F')[testLinks] #transforma a matriz em um vetor linha, começando pelas colunas (Ordem Fortran, igual ao matlab)
            predictions = reshape(predictions,(1,predictions.shape[0]))
            #print(predictions.shape)

            #Obtive os status de pares atuais (link ou nao) em ordem
            #para checar o erro de predições

            real = convergenceScore["D"][2,:].reshape(1,convergenceScore["D"].shape[1])
            npairsTs = convergenceScore["npairs"]

            #Função de avaliação que gera o desempenho do modelo
            newScore, newScoreRmse, flScore, prec, red = evalModelByDoc3(predictions, real, npairsTs)

            print("test auc: {} rmse: {:.5f} fl: {:.4f} \n".format(newScore, newScoreRmse, flScore[0][0]))

        else:
            newScoreTr, newScore, trainError, testError = 0, 0, array([]), array([])

        #mantem o controle de quantas epochs em uma linha tem levado para uma melhoria limitada
        if newScore < lastScore + (1*10**-4): # + 1e-4
            badEpochs = badEpochs +1
        else:
            badEpochs = 0

        #para inicialmente se os parâmetros explodirem / tivermos muitas epochs ruins sucessivas
        if any(isnan(U[:])) or any(isnan(P[:])) or any(isnan(Q[:])) or any(isnan(WBilinear[:])) or isnan(newScore) or (0 and badEpochs > 3):
            U, P, Q = UOld, POld, QOld
            UBias = UBiasOld
            ULatentScaler = ULatentScalerOld
            GLatentScaler = GLatentScalerOld
            WPair = WPairOld
            WBias = WBiasOld
            WBilinear = WBilinearOld

            print("early stopping at epoch {}: auc = {}.4f -> {}.4f \n".format(e,lastScore,newScore))
            break;

        if newScore > bestScore:
            UOld, POld, QOld = U, P, Q
            UBiasOld, ULatentScalerOld = UBias, ULatentScaler
            GLatentScalerOld = GLatentScaler
            WPairOld,WBiasOld = WPair, WBias
            WBilinearOld = WBilinear

            bestScore = newScore

        lastScore = newScore

    #end for externo

    U = U.conj().transpose()
    P = P.conj().transpose()
    Q = Q.conj().transpose()

    return (U, P, Q, UBias, ULatentScaler, GLatentScaler, WPair, WBias, WBilinear)