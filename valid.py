import torch
from utils import Adder
import torch.nn.functional as f
from metrics import calculate_psnr


def _valid(model, args, ep, ots):
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(ots):
            input_img, label_img = data

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = calculate_psnr(p_numpy*255, label_numpy*255, crop_border=0, test_y_channel=True)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
