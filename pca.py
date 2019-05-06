import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('./usps.train')
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
cov_mat = np.cov(X.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
cov_mat = np.cov(X.T)
eigenValues, eigenVectors = np.linalg.eig(cov_mat)
sortedEigenValueIndex =  np.argsort(-eigenValues)
sortedEigenVectors = eigenVectors[:,sortedEigenValueIndex] 
sortedEigenVectors = sortedEigenVectors[:,:16]


cols=sortedEigenVectors.shape[1]

for i in range(0,cols):
	new_mat=np.reshape(sortedEigenVectors[:,i],(16,16))
	plt.gray();
	plt.imshow(new_mat)
	plt.show()

tot = sum(eigenValues)
cum_var_exp_dict = dict()
ind_var_exp_dict = dict()
comp=0
cum_var_exp=0
k70=[]
k80=[]
k90=[]
for i in sorted(eigenValues, reverse=True):
	comp=comp+1
	var_exp = (i / tot)*100
	ind_var_exp_dict[comp]=var_exp
	cum_var_exp+=var_exp
	if cum_var_exp>=70:
		k70.append(comp)
	if cum_var_exp>=80:
		k80.append(comp)
	if cum_var_exp>=90:
		k90.append(comp)
	cum_var_exp_dict[comp]=cum_var_exp
	#comp=comp+1
comps_for_v70=k70[0]
comps_for_v80=k80[0]
comps_for_v90=k90[0]

compsfinal=[]
compsfinal.append(comps_for_v70)
compsfinal.append(comps_for_v80)
compsfinal.append(comps_for_v90)
compsfinal.append(256)


plt.figure()
plt.bar(cum_var_exp_dict.keys(), cum_var_exp_dict.values(),width=2)
plt.bar(ind_var_exp_dict.keys(), ind_var_exp_dict.values(),width=2)
plt.xlabel('Number of components', fontsize=16)
plt.ylabel('explained variance ratio', fontsize=16)
plt.legend(['cumulative explained variance ratio','individual explained variance ratio'], fontsize=16)
plt.axis([1,256, 0,100])
plt.show()

df1 = pd.read_csv('./usps.valid')
X1 = df1.iloc[:,1:].values
y1 = df1.iloc[:,0].values
# X_std_val = StandardScaler().fit_transform(X1)
df2 = pd.read_csv('./usps.test')
X2= df2.iloc[:,1:].values
y2 = df2.iloc[:,0].values
#X_std_tst = StandardScaler().fit_transform(X2)
for i in compsfinal:
	matrix_w = np.hstack(eig_pairs[item][1].reshape(256,1) for item in range(0,i,1))
	proj=X.dot(matrix_w)
	print("For ",i,"PC's")
	for k in [0.0001, 0.001, 0.01, 0.1]:
		clf = SGDClassifier(alpha=k,loss="hinge", penalty="l2")
		clf.fit(proj, y)
		proj_val=X1.dot(matrix_w)
		proj_tst=X2.dot(matrix_w)
		score_val=clf.score(proj_val,y1)
		score_tst=clf.score(proj_tst,y2)
		print("on validation set accuracy for alpha =",k,"is",score_val*100,"%")
		print("on Test set accuracy for alpha =",k,"is",score_tst*100,"%")

#computing for SVM without feature selection
for k in [0.0001, 0.001, 0.01, 0.1]:
	clf = SGDClassifier(alpha=k,loss="hinge", penalty="l2")
	clf.fit(X, y)
	score_tst=clf.score(X2,y2)
	print("on Test set SVM accuracy for alpha =",k,"is",score_tst*100,"%")







