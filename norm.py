import torch
import math
import torch.nn.functional as F
from einops import rearrange
#B, C, H, W = 1, 3, 512, 512
B,C,H,W = 1,1,4,4
x = torch.arange(1, 1 + B*C*H*W).reshape((B, C, H, W)).to(torch.float32).cuda()
x = x.repeat(1, 2, 1, 1)
C=2
print("====================X====================")
print(x.shape)
print(x)
#x = torch.rand((B,C,H,W)).cuda()
kernel_size = 2
stride = 2
patch_size = kernel_size

# Step 1: unfold
patches = F.unfold(x, kernel_size=(patch_size, patch_size), stride=stride)  # (B, C*patch_area, num_patches)
patches = patches.view(B, C, patch_size * patch_size, -1)  # (B, C, patch_area, num_patches)
print("====================unflod_x====================")
print(patches.shape)
print(patches)
print("====================mean====================")
# Step 2: 求均值
mean = patches.mean(dim=2)  # (B, C, num_patches)
print(mean.shape)
print(mean)

print("====================std====================")
std = patches.std(dim=2)    # (B, C, num_patches)
print(std.shape)
print(std)

print("====================out mean====================")
out_mean = mean.view(B,C,H//patch_size, W//patch_size)
print(out_mean.shape)
print(out_mean)
print("====================out mean====================")
out_std = std.view(B,C,H//patch_size, W//patch_size)
print(out_std.shape)
print(out_std)
exit(0)
print("====================repeated====================")
# Step 3: 每个均值复制为 patch_size x patch_size，再 fold 回原图
mean = mean.unsqueeze(2).repeat(1, 1, patch_size * patch_size, 1)  # (B, C, patch_area, num_patches)
mean = mean.view(B, C * patch_size * patch_size, -1)  # (B, C*patch_area, num_patches)
mean = F.fold(mean, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

std = std.unsqueeze(2).repeat(1, 1, patch_size * patch_size, 1)  # 同样处理 std
std = std.view(B, C * patch_size * patch_size, -1)
std = F.fold(std, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
print(mean.shape)
print(mean)
print(std.shape)
print(std)
print("====================acted to X====================")
y = x * std + mean
print(y)

print("====================local_normed_x====================")
normed_x = (x - mean) / (std + 1e-7)
print(normed_x.shape)
print(normed_x)

# print("====================flod_normed_x====================")
# normed_x = normed_x.view(B, -1, C*kernel_size*kernel_size) #(b, num_patches, c*kernel_size*kernel_size)
# normed_x = torch.transpose(normed_x, -2, -1) #(b, c*kernel_size*kernel_size, num_patches)
# normed_x = F.fold(normed_x, output_size=(H, W), kernel_size=(kernel_size,kernel_size), stride=stride)
# print(normed_x.shape)
# print(normed_x)