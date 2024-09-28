import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

image = Image.open('压缩图片.png')
image_array = np.array(image.convert('L'))

# 铺平
image_flat = image_array.reshape(-1, image_array.shape[-1])


plt.figure(figsize=(14, 7))
#原图
plt.subplot(2, 4, 1)
plt.imshow(image_array,cmap='gray')
plt.title('Original Image')
plt.axis('off')

i=1
ncom=[0.01,0.02,0.05,0.1,0.2,0.3]
for n in ncom:
    print(n)
    pca = PCA(n_components=n)
    pca.fit(image_flat)
    image_pca = pca.transform(image_flat)

    # 恢复成原始维度
    image_re = pca.inverse_transform(image_pca)
    print(image_re)
    # 恢复成图像矩阵
    image_re = image_re.reshape(image_array.shape)
    print(image_re)
    if i<4:
        plt.subplot(2,4,i+1)
    else:
        plt.subplot(2,4,i+2)
    plt.imshow(image_re.astype(np.uint8),cmap='gray')
    plt.title(f'Compressed Image ({n*100}%)')
    plt.axis('off')
    i+=1
plt.show()
