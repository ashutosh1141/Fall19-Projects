from sklearn import datasets
import numpy as np
import pandas as pd
import math
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler


def spectral_clustering(x,k):
	#compute symmetric matrix
	points=np.array(x)
	dist_condensed = pdist(points)
	dist = squareform(dist_condensed)
	rows = dist.shape[0]
	cols = dist.shape[1]
	for i in range(0, rows):
		for j in range(0, cols):
			dist[i][j]=math.exp(-15*dist[i][j])
	#compute laplacian
	D= np.zeros(shape=(rows,cols))
	for i in range(0,rows,1):
		row_sum=sum(dist[i])
		D[i][i]=row_sum
	L=D-dist
	#compute eigen_values and vectors of l
	eigenValues, eigenVectors = np.linalg.eig(L)
	sortedEigenValueIndex =  np.argsort(eigenValues)
	sortedEigenVectors = eigenVectors[:,sortedEigenValueIndex] 
	sortedEigenVectors = sortedEigenVectors[:,:k]
	eigenDataFrame = pd.DataFrame(sortedEigenVectors)
	#make clusters
	C=KMeans(n_clusters=k, random_state=0).fit(eigenDataFrame)
	return C.labels_

def img_seg(x,k):
	#compute symmetric matrix
	points=np.array(x)
	dist_condensed = pdist(points)
	dist = squareform(dist_condensed)
	rows = dist.shape[0]
	cols = dist.shape[1]
	for i in range(0, rows):
		for j in range(0, cols):
			dist[i][j]=math.exp(-0.1*dist[i][j])
	#compute laplacian
	D= np.zeros(shape=(rows,cols))
	for i in range(0,rows,1):
		row_sum=sum(dist[i])
		D[i][i]=row_sum
	L=D-dist
	#compute eigen_values and vectors of l
	eigenValues, eigenVectors = sp.linalg.eigh(L,eigvals=(0,k-1))
	sortedEigenValueIndex =  np.argsort(eigenValues)
	sortedEigenVectors = eigenVectors[:,sortedEigenValueIndex] 
	eigenDataFrame = pd.DataFrame(sortedEigenVectors)
	#make clusters
	C=KMeans(n_clusters=k, random_state=0).fit(eigenDataFrame)
	#plot image
	labels=np.reshape(C.labels_,(81,121))
	plt.imshow(labels)
	plt.show()
	return C.labels_

if __name__ == '__main__':

	#generate dataset and plot it
	x,y = datasets.make_circles(n_samples=1500, factor=.5, noise=.05)
	cmap = 'viridis'
	dot_size=50
	plt.scatter(x[:, 0], x[:, 1],c=y,s=dot_size, cmap=cmap)
	plt.show()

	#using Kmeans clustering
	C=KMeans(n_clusters=2, random_state=0).fit(x)
	cmap = 'viridis'
	dot_size=50
	fig, ax = plt.subplots(figsize=(9,7))
	ax.set_title('Kmeans clustered data', fontsize=18, fontweight='demi')
	ax.scatter(x[:, 0], x[:, 1],c=C.labels_,s=dot_size, cmap=cmap)
	plt.show()

	#spectral clustered data
	cls_out=spectral_clustering(x,2)
	cmap = 'viridis'
	dot_size=50
	fig, ax = plt.subplots(figsize=(9,7))
	ax.set_title('Spectral Clustered data', fontsize=18, fontweight='demi')
	ax.scatter(x[:, 0], x[:, 1],c=cls_out,s=dot_size, cmap=cmap)
	plt.show()

	img = cv2.imread('seg.jpg', 0)
	cv2.imshow('image', img)
	pixels=np.reshape(img, (9801, 1))
	img_seg(pixels,2)
	C1=KMeans(n_clusters=2, random_state=0).fit(pixels)
	labels_kmean=np.reshape(C1.labels_,(81,121))
	plt.imshow(labels_kmean)
	plt.show()
