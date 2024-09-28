import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = Image.open("压缩图片.png")
image_array = np.array(image)
# 进行奇异值分解
U, S, Vt = np.linalg.svd(image_array, full_matrices=False)

plt.Figure(figsize=(14,7))
plt.subplot(2,4,1)
plt.imshow(image_array)
plt.title('Original Image')
plt.axis('off')

ncom=[0.01,0.02,0.05,0.1,0.2,0.3]
i=1
for n in ncom:
    compression_ratio = n
    k = int(compression_ratio * min(image_array.shape))
    # 重构图像
    compressed_image_array = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    compressed_image = Image.fromarray(compressed_image_array)
    if i<4:
        plt.subplot(2, 4, i+1)
    else:
        plt.subplot(2,4,i+2)
    plt.imshow(compressed_image)
    plt.title(f'Compressed Image({n*100}%)')
    plt.axis('off')
    i+=1
plt.suptitle('Compressed of Various Radios')
plt.show()