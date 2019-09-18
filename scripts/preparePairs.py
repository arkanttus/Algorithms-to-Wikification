from numpy import *
from createPairs import *
from tqdm import tqdm
from time import *

scaling = 1

print("Lendo todos os ids samples... \n")

file1 = open("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/codes-to-create-school-wiki2014/5132_wss_wiki2014_sample-ids.txt", 'r')
In = loadtxt(file1, delimiter=',', dtype='int')
file1.close()

docs = In.reshape((In.shape[0],1)) #In[:,0]
n = docs.shape[0]

print("Lendo todos rates samples...\n")
file = open("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/codes-to-create-school-wiki2014/5132_wss_wiki2014_sample-rates.txt", 'r', encoding='utf-8')
In = genfromtxt(file, encoding='utf-8', delimiter='\t', dtype=None)
file.close()

rates = In.transpose()['f2'].reshape((In.shape[0],1)) #In[:,2]

print("Lendo características de nó...\n")
file = open("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/Italo-colecao-wiki/nodes_features_new.txt", 'r')
In = genfromtxt(file,encoding='utf-8', skip_header=1, delimiter=';', dtype=None)
x1 = In['f2'].reshape((In.shape[0],1)) #inlink ratio (new attribute)
x2 = In['f3'].reshape((In.shape[0],1)) #outlink ratio (new attribute)
x3 = In['f4'].reshape((In.shape[0],1)) #generality
file.close()

titles = In['f1'].reshape((In.shape[0],1)) #title names

print("Lendo características de pares...\n")
#file2 = open("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/Italo-colecao-wiki/pairs_raw_MW2013+NF_1.csv", 'r')
file2 = open("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/Italo-colecao-wiki/pairs_raw.npy", 'rb')
#In = loadtxt(file2, skiprows=1, encoding='utf-8', delimiter=',', dtype={'names':('f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13'),'formats':('int16','int16','float16','float16','float16','float16','float16','float16','float16','float16','float16','float16','int16','int16')})
In = load(file2)
lins = In.shape[0]
print("\n")
print(In)
print("\n")
print(In.dtype)
print("\n")
print(In.shape)
In = In.transpose()
file2.close()
'''
f0 = double(In[:,2])   #relatedness
f1 = double(In[:,3])   #relatednessToContext
f2 = double(In[:,4])   #occurance
f3 = double(In[:,5])   #avgLinkProb
f4 = double(In[:,6])   #maxLinkProb
f5 = double(In[:,7])   #firstOcurr
f6 = double(In[:,8])   #lastOcurr
f7 = double(In[:,9])  #spread
f8 = double(In[:,10])  #avgDisambigConf
f9 = double(In[:,11])  #maxDisambigConf
f10 = double(In[:,12]) #is there a label?
f11 = double(In[:,13]) #link status
'''

f0 = double(In['f2'].reshape((lins,1)))   #relatedness
f1 = double(In['f3'].reshape((lins,1)))   #relatednessToContext
f2 = double(In['f4'].reshape((lins,1)))   #occurance
f3 = double(In['f5'].reshape((lins,1)))   #avgLinkProb
f4 = double(In['f6'].reshape((lins,1)))   #maxLinkProb
f5 = double(In['f7'].reshape((lins,1)))   #firstOcurr
f6 = double(In['f8'].reshape((lins,1)))   #lastOcurr
f7 = double(In['f9'].reshape((lins,1)))   #spread
f8 = double(In['f10'].reshape((lins,1)))  #avgDisambigConf
f9 = double(In['f11'].reshape((lins,1)))  #maxDisambigConf
f10 = double(In['f12'].reshape((lins,1))) #is there a label?
f11 = double(In['f13'].reshape((lins,1))) #link status

print(In)
print(In.shape)
print(f11)
print(f11.shape)

print("Obtendo informação topológica...\n")
A = zeros((n,n))
#obtain topology information...
for i in range(0,n):
    A[i,:] = f11[ (n * (i+1) - n) : (n * (i+1)), 0 ]

D = A

K = zeros((n,n))
for i in range(0,n):
    K[i,:] = f10[ (n * (i+1)) - n : (n * (i+1)), 0 ]

# make normalization by standard deviation of each attribute
if scaling:

    print("Scaling node features...\n")

    #nodes atributes
    F   = block([x1,x2,x3])
    mF = mean(F,axis=0)
    sF = std(F,ddof=1,axis=0)

    for j in range(0, F.shape[1] ):
        F[:,j] = ( F[:,j] - mF[j] / sF[j] )

    F = single(F.conj().transpose()) #use generality as node feature

    #dyads attributes
    M = concatenate((f0,f1,f2,f3,f4,f5,f6,f7,f8,f9), axis=1)

    #obtain mean and std of each attribute (column)
    mM = mean(M, axis=0)
    sM = std(M, ddof=1, axis=0)

    #make normalization by std
    #subtract the mean of each attribute and divide by std
    for j in range(0, M.shape[1]):
        M[:,j] = (M[:,j] - mM[j]) / sM[j]

    print("Scaling pair features...\n")
    f = M.shape[1]
    #X = single( tile(0.0, concatenate( (array([n]), f, array([n])) )) )
    X = single( tile(0.0, block([n, f, n])))

    for k in range(0,f):
        for i in range(0,n):
            X[:,k,i] = M[ (n * (i+1)) - n : (n * (i+1)) , k]

    print("Generating Dbin, Z and J pairs...\n")

    Dbin, Z = createPairs(D,K);

    print("Saving all variables required...\n")

    savez("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/5132_loose_wss-wiki2014_vfinal.npz", docs=docs, titles=titles, Dbin=Dbin, X=X, F=F, Z=Z, rates=rates)

    print("File saved with successful\n")

#end If Scaling




