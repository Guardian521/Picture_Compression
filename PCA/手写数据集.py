from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
data=load_digits()
X=data.data
y=data.target
df=pd.DataFrame(data=X,columns=data.feature_names)
df['target']=y
#划分数据集
x_train,x_test,y_trian,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
#训练
svm=SVC()
svm.fit(x_train,y_trian)
accuracy = svm.score(x_test, y_test)
print("Accuracy:", accuracy)
#pca降维后的svm分类结果
pca=PCA(n_components=0.80)
X_pca = pca.fit_transform(X)
x_train_p,x_test_p,y_trian_p,y_test_p=train_test_split(X_pca,y,test_size=0.2,random_state=3)
svm.fit(x_train_p,y_trian_p)
accuracy_pca=svm.score(x_test_p,y_test_p)
print("Accuracy_PCA:",accuracy_pca)



