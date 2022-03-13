# Photo2Sketch(Learn to Sketch: A fast approach for universal photo sketch)

By Zhi-Song Liu, Wan-Chi Siu and H. Anthony Chan

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We propose a joint photo to sketch and sketch to photo convolutional neural network

Please check our [paper](https://ieeexplore.ieee.org/document/9689529)

# BibTex

       @INPROCEEDINGS{9689529,
        author={Liu, Zhi -Song and Siu, Wan-Chi and Chan, H. Anthony},
        booktitle={2021 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
        title={Learn to Sketch: A fast approach for universal photo sketch}, 
        year={2021},
        volume={},
        number={},
        pages={1450-1457},
        doi={}}
        
# For proposed model, we claim the following points:

• We implicitly discover mapping correlation between photos and sketches by forming a loop to achieve self supervision. That is, we use Photo2Sketch to map photos to corresponding sketches and use Sketch2Photo to map sketches back to photos.

• To generate photo-realistic sketches, we introduce reference based training losses to encourage the generated sketches close to general sketches.

• Furthermore, we introduce the soft weighted function to adaptively generate the sketch results so that we can assign confidences to different drawing lines. Hence, multiple sketch candidates can be generated.

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch >= 1.10
    NVIDIA GPU + CUDA

# Complete Architecture
The complete architecture is shown as follows,

![network](/figure/network.png)

# Implementation
## 1. Quick testing
---------------------------------------
1. Copy your image to folder "Test" and run 
```sh
$ python eval_test.py
```
The output images will be in folder "Result"
2. For multiple results, run
```sh
$ python eval_multiple.py
```

## 2. Training
---------------------------
### s1. Download the training images from COCO.
    
https://cocodataset.org/#home

### s2. Download the sketch references.
https://github.com/HaohanWang/ImageNet-Sketch

   
### s3. Start training on Pytorch
1. Train the Denoising VAE by running
```sh
$ python main_sketch.py
```

---------------------------

## Partial image visual comparison

## 1. Visualization comparison
Results on photo to sketch

![figure2](/figure/teaser.png)
![figure3](/figure/compare.png)


# Reference
You may check our newly work on [Multiple style transfer via variational autoencoder](https://github.com/Holmes-Alan/ST-VAE)

