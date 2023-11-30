# TensorBot-Vision


## Pipeline
![image](./images/pipeline.jpeg)



- Turn off firewall
```
sudo ufw disable
```




## Install mmdetect
```
# GPU
conda install pytorch torchvision -c pytorch


# CPU
conda install pytorch torchvision cpuonly -c pytorch

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"


git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

```

### Verify the installation
```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```



## References
- https://towardsdatascience.com/what-i-was-missing-while-using-the-kalman-filter-for-object-tracking-8e4c29f6b795
- https://mmdetection.readthedocs.io/en/latest/get_started.html
