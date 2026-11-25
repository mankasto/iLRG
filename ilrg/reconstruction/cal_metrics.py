import os
import cv2
import lpips
import torch
import numpy as np

from .metrics import psnr


def get_images(images_list, img_size=28):
    images = []
    for image_path in images_list:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (img_size, img_size))
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        images.append(image)
    images = torch.stack(images).to(device)
    return images


if __name__ == '__main__':
    device = 'cuda:0'
    dataset = 'mnist'
    lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
    reconstruction_dir = os.path.join('assets', 'reconstructions')
    x_dir = os.path.join(reconstruction_dir, f'{dataset}_gt')
    y_dir = os.path.join(reconstruction_dir, f'{dataset}_ig')
    z_dir = os.path.join(reconstruction_dir, f'{dataset}_ours')
    x = [os.path.join(x_dir, filename) for filename in os.listdir(x_dir)]
    y = [os.path.join(y_dir, filename) for filename in os.listdir(y_dir)]
    z = [os.path.join(z_dir, filename) for filename in os.listdir(z_dir)]
    x = get_images(x, 28 if dataset == 'mnist' else 32)
    y = get_images(y, 28 if dataset == 'mnist' else 32)
    z = get_images(z, 28 if dataset == 'mnist' else 32)
    with torch.no_grad():
        lpips_score = lpips_loss(y, x).squeeze().mean()
        lpips_score2 = lpips_loss(z, x).squeeze().mean()
        psnr1 = psnr(y, x)
        psnr2 = psnr(z, x)
    print('LPIPS: IG-%.2f; Ours-%.2f' % (lpips_score.item(), lpips_score2.item()))
    print('PSNR: IG-%.2f; Ours-%.2f' % (psnr1, psnr2))
