import torch

# def spiralArray(H, W):
#     DIRS = (0, 1), (1, 0), (0, -1), (-1, 0)  # 右下左上
#     ans = torch.zeros((H, W), dtype=torch.long)
#     flag = torch.zeros((H, W), dtype=torch.bool)
#     i = j = di = 0  # 初始位置和方向
#     n = 0  # 当前填充的数字，从0开始
#     for _ in range(H * W):
#         ans[i][j] = n
#         n += 1
#         flag[i][j] = True  # 标记当前位置已访问
#         x, y = i + DIRS[di][0], j + DIRS[di][1]  # 下一个位置
#         if x < 0 or x >= H or y < 0 or y >= W or flag[x][y]:  # 如果越界或已访问
#             di = (di + 1) % 4  # 改变方向
#         i += DIRS[di][0]
#         j += DIRS[di][1]
#     return ans
# 螺旋顺序遍历生成索引
def generate_spiral_indices(h, w):
    matrix = torch.zeros((h, w), dtype=torch.long)
    result = []
    top, bottom, left, right = 0, h-1, 0, w-1

    while top <= bottom and left <= right:
        # 从左到右遍历上边
        for i in range(left, right + 1):
            result.append((top, i))
        top += 1

        # 从上到下遍历右边
        for i in range(top, bottom + 1):
            result.append((i, right))
        right -= 1

        if top <= bottom:
            # 从右到左遍历下边
            for i in range(right, left - 1, -1):
                result.append((bottom, i))
            bottom -= 1

        if left <= right:
            # 从下到上遍历左边
            for i in range(bottom, top - 1, -1):
                result.append((i, left))
            left += 1

    # 将生成的坐标转为一维索引
    indices = [i * w + j for i, j in result]
    return indices


b, c, h, w = 2, 2, 4, 3
total_elements = b * c * h * w
sequence = torch.arange(1, total_elements + 1, dtype=torch.float32)
x = sequence.view(b, c, h, w)
print(x)

spiral_indices = generate_spiral_indices(h, w)
print(f"螺旋顺序索引: {spiral_indices}")

indices = torch.tensor(spiral_indices)
flattened_tensor = x.view(b, c, -1)
result = flattened_tensor[:, :, indices]

print("螺旋遍历结果:")
print(result)

inv = torch.flip(result,dims=[-1])
print("逆序螺旋遍历结果:")
print(inv)

restored_tensor = torch.zeros((b, c, h, w), dtype=torch.float32)
indices = torch.tensor(spiral_indices, dtype=torch.long)
restored_tensor.view(b, c, -1).scatter_(2, indices.unsqueeze(0).unsqueeze(0).expand(b, c, -1), result)
print("螺旋复原结果:")
print(restored_tensor.shape) 
print(restored_tensor)

restored_inv = torch.zeros((b, c, h, w), dtype=torch.float32)
indices = torch.tensor(spiral_indices, dtype=torch.long)
restored_inv.view(b, c, -1).scatter_(2, indices.unsqueeze(0).unsqueeze(0).expand(b, c, -1), inv.flip([-1]))
print("逆序螺旋复原结果:")
print(restored_inv.shape) 
print(restored_inv)