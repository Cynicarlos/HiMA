import numpy as np
import matplotlib.pyplot as plt
import os
import re
import rawpy

def pack_raw(raw):
    # pack Bayer image to 4 channels (RGBG)
    im = raw.raw_image_visible.astype(np.uint16)
    H, W = im.shape
    im = np.expand_dims(im, axis=0)
    out = np.concatenate((im[:, 0:H:2, 0:W:2],
                        im[:, 0:H:2, 1:W:2],
                        im[:, 1:H:2, 1:W:2],
                        im[:, 1:H:2, 0:W:2]), axis=0)
    return out

black_level = 512
white_level = 16383
# black_level = 64
# white_level = 1023

#input_path = 'E:\Deep Learning\datasets\RAW\SID\Sony\Sony\short/00086_00_0.1s.ARW'
input_path = 'E:\Deep Learning\datasets\RAW\SID\Sony\Sony\short/00110_00_0.1s.ARW'
#input_path =  'C:/Users/Carlos/Desktop/low.DNG'
input_raw = rawpy.imread(input_path)
#input_raw = pack_raw(input_raw)#(4,2848/2,4256/2) numpy
input_raw = input_raw.raw_image_visible.astype(np.uint16)

#gt_path = 'E:\Deep Learning\datasets\RAW\SID\Sony\Sony\long/00086_00_30s.ARW'
gt_path = 'E:\Deep Learning\datasets\RAW\SID\Sony\Sony\long/00110_00_30s.ARW'
#gt_path = 'C:/Users/Carlos/Desktop/normal.DNG'
gt_raw = rawpy.imread(gt_path)

#gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy
#gt_raw = pack_raw(gt_raw)#(4,2848/2,4256/2) numpy
gt_raw = gt_raw.raw_image_visible.astype(np.uint16)


input_raw = (np.float32(input_raw) - black_level) / np.float32(white_level - black_level)
gt_raw = (np.float32(gt_raw) - black_level) / np.float32(white_level - black_level)
#gt_rgb = np.float32(gt_rgb) / np.float32(65535)

gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)
fixed_ratioed_raw = input_raw * 300
fixed_ratioed_raw = np.maximum(np.minimum(fixed_ratioed_raw, 1.0), 0.0)

gt_raw_mean = np.mean(gt_raw)
input_raw_mean = np.mean(input_raw)
# gt_raw_std = np.std(gt_raw)
# input_raw_std = np.std(input_raw)
ratio = gt_raw_mean / input_raw_mean
global_mean_raw = input_raw * ratio
global_mean_raw = np.maximum(np.minimum(global_mean_raw, 1.0), 0.0)

print(gt_raw.shape, fixed_ratioed_raw.shape, global_mean_raw.shape)
print(gt_raw.mean(), fixed_ratioed_raw.mean(), global_mean_raw.mean())

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
cmap = 'hot'  # 选择颜色映射
im1 = axes[0,0].imshow(fixed_ratioed_raw, cmap=cmap, interpolation='nearest')
fig.colorbar(im1, ax=axes[0,0])  # 添加颜色条
axes[0,0].set_title('Fixed Global Ratio')

im2 = axes[0,1].imshow(global_mean_raw, cmap=cmap, interpolation='nearest')
fig.colorbar(im2, ax=axes[0,1])  # 添加颜色条
axes[0,1].set_title('Global Ratio')

local_aligned_raw = input_raw.copy()
local_aligned_raw_with_std = input_raw.copy()
H, W = input_raw.shape
p_w, p_h = 4, 4  # patch的宽度和高度
n_h = H // p_h  # 水平方向上的patch数量
n_w = W // p_w  # 垂直方向上的patch数量
# patches = input_raw.reshape(n_h, p_h, n_w, p_w).swapaxes(1, 2)
# gt_raw_patches = gt_raw.reshape(n_h, p_h, n_w, p_w).swapaxes(1, 2)
patches = input_raw[:n_h*p_h, :n_w*p_w].reshape(n_h, p_h, n_w, p_w).swapaxes(1, 2)
gt_raw_patches = gt_raw[:n_h*p_h, :n_w*p_w].reshape(n_h, p_h, n_w, p_w).swapaxes(1, 2)
mean_values = patches.mean(axis=(2, 3))
std_values = patches.std(axis=(2, 3))
patch_gt_raw_mean_values = gt_raw_patches.mean(axis=(2, 3))
patch_gt_raw_std_values = gt_raw_patches.std(axis=(2, 3))
for i in range(n_h):
    for j in range(n_w):
        t = input_raw[i*p_h:(i+1)*p_h, j*p_w:(j+1)*p_w]
        #t *= ratio
        v = t * (patch_gt_raw_mean_values[i, j] / (mean_values[i, j]+1e-6))
        local_aligned_raw[i*p_h:(i+1)*p_h, j*p_w:(j+1)*p_w] = np.maximum(np.minimum(v, 1.0), 0.0)

        #z = (std(y)/std(x))*x + (miu(y)-(std(y)/std(x))*miu(x))
        c = patch_gt_raw_std_values[i,j] / (std_values[i,j]+1e-6) #std(y)/std(x)
        t = c * t + (patch_gt_raw_mean_values[i, j] - c * mean_values[i, j])
        local_aligned_raw_with_std[i*p_h:(i+1)*p_h, j*p_w:(j+1)*p_w] = np.maximum(np.minimum(t, 1.0), 0.0)

im3 = axes[1,0].imshow(local_aligned_raw, cmap=cmap, interpolation='nearest')
fig.colorbar(im3, ax=axes[1,0])  # 添加颜色条
axes[1,0].set_title(f'Local Ratio')

# im4 = axes[1,1].imshow(gt_raw, cmap=cmap, interpolation='nearest')
# fig.colorbar(im4, ax=axes[1,1])  # 添加颜色条
# axes[1,1].set_title('GT_RAW')

im4 = axes[1,1].imshow(local_aligned_raw_with_std, cmap=cmap, interpolation='nearest')
fig.colorbar(im4, ax=axes[1,1])  # 添加颜色条
axes[1,1].set_title('Local Ratio with STD')

plt.tight_layout()
plt.show()

# plt.imsave('Fixed_Global_Ratio.png', fixed_ratioed_raw, cmap='hot')
# plt.imsave('Global_Ratio.png',       global_mean_raw,   cmap='hot')
#plt.imsave('Local_Ratio_4.png',        local_aligned_raw, cmap='hot')
# plt.imsave('GT_RAW.png',             gt_raw,            cmap='hot')
# plt.imsave('Local_Ratio_STD_4.png',    local_aligned_raw_with_std, cmap='hot')
#plt.imsave('Input_RAW.png', np.maximum(np.minimum(input_raw, 1.0), 0.0), cmap='hot')
exit(0)


flattened_input_raw = input_raw.flatten()
flattened_fixed_ratioed_raw = fixed_ratioed_raw.flatten()
flattened_ratioed_raw = ratioed_raw.flatten()
flattened_patch_ratioed_raw = patch_ratioed_raw.flatten()
flattened_patch_ratioed_raw_with_std = patch_ratioed_raw_with_std
flattened_gt_raw = gt_raw.flatten()
bins = 200

# flattened_data1 = flattened_data1[(flattened_data1 < 0.01)]
# flattened_data2 = flattened_data2[(flattened_data2 < 0.01)]
# flattened_data3 = flattened_data3[(flattened_data3 < 0.01)]
# flattened_data4 = flattened_data4[(flattened_data4 < 0.01)]
# flattened_data5 = flattened_data5[(flattened_data5 < 0.01)]
#plt.hist(flattened_data1, bins=bins, range=(0, 0.01), edgecolor='black', alpha=0.5, label='RAW', color='blue')
#plt.hist(flattened_data2, bins=bins, range=(0, 0.01), edgecolor='black', alpha=0.5, label='Fixed_Global_Ratio', color='red')
#plt.hist(flattened_data3, bins=bins, range=(0, 0.01), edgecolor='black', alpha=0.5, label='Global_Ratio', color='red')
# plt.hist(flattened_data4, bins=bins, range=(0, 0.01), edgecolor='black', alpha=0.5, label='Patched_Ratio', color='red')
# plt.hist(flattened_data5, bins=bins, range=(0, 0.01), edgecolor='black', alpha=0.5, label='GT_RAW', color='green')

flattened_input_raw = flattened_input_raw[(flattened_input_raw > 0.2) & (flattened_input_raw < 0.6)]
flattened_fixed_ratioed_raw = flattened_fixed_ratioed_raw[(flattened_fixed_ratioed_raw > 0.2) & (flattened_fixed_ratioed_raw < 0.6)]
flattened_ratioed_raw = flattened_ratioed_raw[(flattened_ratioed_raw > 0.2) & (flattened_ratioed_raw < 0.6)]
flattened_patch_ratioed_raw = flattened_patch_ratioed_raw[(flattened_patch_ratioed_raw > 0.2) & (flattened_patch_ratioed_raw < 0.6)]
flattened_patch_ratioed_raw_with_std = flattened_patch_ratioed_raw_with_std[(flattened_patch_ratioed_raw_with_std > 0.2) & (flattened_patch_ratioed_raw_with_std < 0.6)]
flattened_gt_raw = flattened_gt_raw[(flattened_gt_raw > 0.2) & (flattened_gt_raw < 0.6)]
#plt.hist(flattened_input_raw, bins=bins, range=(0.2, 0.6), edgecolor='black', alpha=0.5, label='RAW', color='blue')
#plt.hist(flattened_fixed_ratioed_raw, bins=bins, range=(0.2, 0.6), edgecolor='black', alpha=0.5, label='Fixed_Global_Ratio', color='red')
#plt.hist(flattened_ratioed_raw, bins=bins, range=(0.2, 0.6), edgecolor='black', alpha=0.5, label='Global_Ratio', color='red')
#plt.hist(flattened_patch_ratioed_raw, bins=bins, range=(0.2, 0.6), edgecolor='black', alpha=0.5, label='Patched_Ratio', color='red')
plt.hist(flattened_patch_ratioed_raw_with_std, bins=bins, range=(0.2, 0.6), edgecolor='black', alpha=0.5, label='Patched_Ratio_STD', color='red')
plt.hist(flattened_gt_raw, bins=bins, range=(0.2, 0.6), edgecolor='black', alpha=0.5, label='GT_RAW', color='green')

# flattened_data1 = flattened_data1[(flattened_data1 > 0.6) & (flattened_data1 < 0.9)]
# flattened_data2 = flattened_data2[(flattened_data2 > 0.6) & (flattened_data2 < 0.9)]
# flattened_data3 = flattened_data3[(flattened_data3 > 0.6) & (flattened_data3 < 0.9)]
# flattened_data4 = flattened_data4[(flattened_data4 > 0.6) & (flattened_data4 < 0.9)]
# flattened_data5 = flattened_data5[(flattened_data5 > 0.6) & (flattened_data5 < 0.9)]
#plt.hist(flattened_data1, bins=bins, range=(0.6, 0.9), edgecolor='black', alpha=0.5, label='RAW', color='blue')
#plt.hist(flattened_data2, bins=bins, range=(0.6, 0.9), edgecolor='black', alpha=0.5, label='Fixed_Global_Ratio', color='red')
#plt.hist(flattened_data3, bins=bins, range=(0.6, 0.9), edgecolor='black', alpha=0.5, label='Global_Ratio', color='red')
# plt.hist(flattened_data4, bins=bins, range=(0.6, 0.9), edgecolor='black', alpha=0.5, label='Patched_Ratio', color='red')
# plt.hist(flattened_data5, bins=bins, range=(0.6, 0.9), edgecolor='black', alpha=0.5, label='GT_RAW', color='green')

# flattened_data1 = flattened_data1[flattened_data1 > 0.9]
# flattened_data2 = flattened_data2[flattened_data2 > 0.9]
# flattened_data3 = flattened_data3[flattened_data3 > 0.9]
# plt.hist(flattened_data1, bins=bins, range=(0.9, 1), edgecolor='black', alpha=0.5, label='RAW', color='blue')
# plt.hist(flattened_data2, bins=bins, range=(0.9, 1), edgecolor='black', alpha=0.5, label='RAW*Ratio', color='red')
# plt.hist(flattened_data3, bins=bins, range=(0.9, 1), edgecolor='black', alpha=0.5, label='GT_RAW', color='green')
############################################preprecessed############################################


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Values in Two Arrays')
plt.legend()
plt.show()