# srgan-tensorflow2.0
SRGAN implementation using Tensorflow 2.0


Tải các folder còn lại (samples, models, DIV2K, archive) tại:

https://drive.google.com/drive/folders/1-C3y8Ae_4NbFjrymxfF2BExlhtCzfs-2?usp=sharing

## Cách sử dụng:

1. Sửa các thông số trong `config.py`
2. Cách chạy file `train.py`

- Warm-up Generator:
```
python train.py --mode=warmup
```
- Bắt đầu train từ đầu
```
python train.py
```
-  Train tiếp:
```
python train.py --mode=continue 
```
- Gen ảnh làm nét:
```python
python train.py --mode=evaluate 
```

3. Losses và các điểm PSNR, SSIM được tính toán trong file `evaluation.ipynb`
4. Triển khai trên web trong thư mục `deployment`

