import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
from models.model import build_net
import time

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # (1, 3, H, W)

def save_image(tensor, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'res.jpg')
    tensor = torch.clamp(tensor, 0, 1)
    tensor += 0.5 / 255
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(save_path)
    print(f"Result saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Single Image Inference Demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the test image')
    parser.add_argument('--save_dir', type=str, default='./demo_output', help='Directory to save the result image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_net()
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_img = load_image(args.image_path).to(device)

    factor = 32
    _, _, h, w = input_img.shape
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    pad_h = H - h
    pad_w = W - w
    input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')

    torch.cuda.synchronize()
    with torch.no_grad():
        start = time.time()
        pred = model(input_img)[2]
        pred = pred[:, :, :h, :w]
        print("Inference Time: {:.4f} seconds".format(time.time() - start))
    torch.cuda.synchronize()

    save_image(pred, args.save_dir)

if __name__ == '__main__':
    main()
