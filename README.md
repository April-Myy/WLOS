# WLOS: Wavelet-based Learning and Optimized Sampling for Image Deraining

## Training

To train the model, use the following command. This command utilizes distributed training on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --use_env --master_port [num] main.py --model_name rain_100L --mode train --num_epoch 800 --data_dir ./Datasets/rain/rain_100L --learning_rate 1e-3 --save_freq 1 --valid_freq 1 --batch_size 4 --num_worker 2 --img_size 256
```


## Testing

To test the trained model, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name rain_100L --mode test --data_dir ./Datasets/rain/rain_100L --test_model ./pkl/rain_100.pkl --save_image True
```


## Inference

For quick inference using the trained model, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path ./pkl/rain_100.pkl --image_path ./Datasets/rain/rain_100L/test/img/rain-001.png --save_dir ./demo_output
```


## Evaluation

After training and testing the model, you can evaluate the performance of the rain streak removal on the test images. We follow previous work by evaluating the model in the YCbCr color space. The evaluation code is provided in the MATLAB script `metrics_matlab.m`, which computes the PSNR and SSIM metrics in the YCbCr color space.