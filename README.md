# srgan-tensorflow2.0
SRGAN re-implement using Tensorflow 2.0

Tải các folder còn lại (samples, models, DIV2K, archive) tại:

https://drive.google.com/drive/folders/1-C3y8Ae_4NbFjrymxfF2BExlhtCzfs-2?usp=sharing

Cách sử dụng:

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

