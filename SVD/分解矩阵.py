import numpy as np
A = np.array([[1, -1], [-2, 2], [2, -2]])

# 完全奇异值分解
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# 紧奇异值分解
U_compact, S_compact, Vt_compact = np.linalg.svd(A, full_matrices=False)

# 打印结果
print("完全奇异值分解：")
print("U:")
print(U)
print("S:")
print(np.diag(S))
print("V^T:")
print(Vt)

print("\n紧奇异值分解：")
print("U_compact:")
print(U_compact)
print("S_compact:")
print(np.diag(S_compact))
print("V^T_compact:")
print(Vt_compact)