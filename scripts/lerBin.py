from numpy import *

arr = load("C:/Users/italo/Documents/pibic/Material Italo - v1/colecao/5132_loose_wss-wiki2014_vfinal.npz")
print(arr.files)
dbin = arr['Dbin']
Z = arr['Z']
X = arr['X']
F = arr['F']
#print(arr['Z'])

#print(arr['Z'].shape)
print("\n")
print(Z)
print(X)
print(X.shape)
#print(arr.transpose())