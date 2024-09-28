from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data=fetch_lfw_people()
X=data.data
y=data.target
df=pd.DataFrame(data=X)
df['target']=y
print(df)
#划分数据集
x_train,x_test,y_trian,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
svm=SVC()
svm.fit(x_train,y_trian)
accuracy = svm.score(x_test, y_test)
print("Accuracy:", accuracy)
#pca降维后的svm分类结果
pca=PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
x_train_p,x_test_p,y_trian_p,y_test_p=train_test_split(X_pca,y,test_size=0.2,random_state=3)
svm.fit(x_train_p,y_trian_p)
accuracy_pca=svm.score(x_test_p,y_test_p)
print("Accuracy_PCA:",accuracy_pca)
#绘制前十二个贡献人脸图
components = pca.components_
plt.figure(figsize=(12, 6))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(components[i].reshape(data.images[0].shape), cmap='gray')
    plt.title(f'Component {i+1}')
    plt.axis('off')
plt.suptitle('Top 12 Components')
plt.show()