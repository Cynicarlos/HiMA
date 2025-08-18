import numpy as np
import torch

def fft(x, D=50):
    f = torch.fft.fftn(x, dim=(-2, -1))
    fshift = torch.fft.fftshift(f, dim=(-2, -1))

    B, C, H, W = x.shape
    x = torch.arange(W, device=x.device) - W // 2
    y = torch.arange(H, device=x.device) - H // 2
    x, y = torch.meshgrid(x, y, indexing='xy')

    gaussian_filter = torch.exp(-(x ** 2 + y ** 2) / (2 * D ** 2)).to(x.device)
    gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    low_freq = fshift * gaussian_filter
    high_freq = fshift * (1 - gaussian_filter)

    low_freq = torch.real(low_freq)
    high_freq = torch.real(high_freq)
    
    return low_freq, high_freq #(b,c_in,h,w)

def inv_fft(f):
    f = torch.fft.ifftn(torch.fft.ifftshift(f, dim=(-2, -1)), dim=(-2, -1))
    f = torch.abs(f)
    return f #(b,c_in,h,w)

def get_low_high_frequency(x, D=50):
    f = torch.fft.fftn(x, dim=(-2, -1))
    fshift = torch.fft.fftshift(f, dim=(-2, -1))

    B, C, H, W = x.shape
    x = torch.arange(W, device=x.device) - W // 2
    y = torch.arange(H, device=x.device) - H // 2
    x, y = torch.meshgrid(x, y, indexing='xy')

    gaussian_filter = torch.exp(-(x ** 2 + y ** 2) / (2 * D ** 2)).to(x.device)
    gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    low_freq = fshift * gaussian_filter
    high_freq = fshift * (1 - gaussian_filter)

    low_freq = torch.fft.ifftn(torch.fft.ifftshift(low_freq, dim=(-2, -1)), dim=(-2, -1))
    high_freq = torch.fft.ifftn(torch.fft.ifftshift(high_freq, dim=(-2, -1)), dim=(-2, -1))

    low_freq = torch.abs(low_freq)
    high_freq = torch.abs(high_freq)

    return low_freq, high_freq #(b,c_in,h,w)

def crop_tow_patch(x):
    _, _, H, W = x.shape
    res = [x[:, :, :, :W//2], x[:, :, :, W//2:]]
    return res

def crop_four_patch(x):
    _, _, H, W = x.shape
    res = [x[:, :, :H//2, :W//2], x[:, :, :H//2, W//2:], x[:, :, H//2:, :W//2], x[:, :, H//2:, W//2:]]
    return res

def crop_six_patch(input_raw):
    _, _, H, W = input_raw.shape
    input_raws = [input_raw[:, :, :H//2, :W//3], input_raw[:, :, :H//2, W//3 : 2*W//3], input_raw[:, :, :H//2, 2*W//3:], input_raw[:, :, H//2:, :W//3], input_raw[:, :, H//2:, W//3 : 2*W//3], input_raw[:, :, H//2:, 2*W//3:]]
    return input_raws

def auto_crop(input_raw, num_row_patch, num_col_patch):
    _, _, H, W = input_raw.shape
    patch_size = patch_size
    input_raws = []
    for i in range(num_col_patch): #0,1
        for j in range(num_row_patch): #0,1,2,3
            start_row = i * patch_size
            if i == num_col_patch - 1:
                end_row = H
            else:
                end_row = start_row + patch_size

            start_col = j * patch_size
            if j == num_row_patch - 1:
                end_col = W
            else:
                end_col = start_col + patch_size
            input_raw_patch = input_raw[:, :, start_row:end_row, start_col:end_col]
            input_raws.append(input_raw_patch)
    return input_raws
    
def auto_reconstruct(input_raws, num_row_patch, num_col_patch, patch_size, full_resolution=(1, 4, 1424, 2128)):
    B, C ,H, W = full_resolution
    reconstructed = torch.zeros((B, C, H, W), dtype=input_raws[0].dtype, device=input_raws[0].device)
    for i in range(num_col_patch):
        for j in range(num_row_patch):
            start_row = i * patch_size
            if i == num_col_patch - 1:
                end_row = H
            else:
                end_row = start_row + patch_size

            start_col = j * patch_size
            if j == num_row_patch - 1:
                end_col = W
            else:
                end_col = start_col + patch_size
            reconstructed[:, start_row:end_row, start_col:end_col] = input_raws[i * num_row_patch + j]
    
    return reconstructed