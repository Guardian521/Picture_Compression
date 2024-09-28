from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
#读取数据
data = load_iris()
X = data.data
y = data.target
df = pd.DataFrame(data=X, columns=data.feature_names)
df['target'] = y
#pca算法，特征值设置为3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
#建立三维图像
for target, color in zip(targets, colors):
    indices = df['target'] == target
    ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], c=color, s=50)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(targets, loc='best')
ax.set_title('PCA of iris')
plt.show()


