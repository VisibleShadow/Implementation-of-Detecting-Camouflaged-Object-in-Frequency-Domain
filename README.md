# Detecting Camouflaged Object in Frequency Domain

An unofficial implementation for *Detecting Camouflaged Object in Frequency Domain, CVPR 2022* in PyTorch.

[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhong_Detecting_Camouflaged_Object_in_Frequency_Domain_CVPR_2022_paper.pdf) | [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhong_Detecting_Camouflaged_Object_CVPR_2022_supplemental.pdf).

***

## Requirements

Here, we list our environment for the experiment on both *Linux* or *Window*.

```
# install

python 3.6
torch == 1.3.1
torchvision == 0.4.2
torch-dct == 0.1.5
numpy == 1.19.5
einops == 0.4.1

# for evaluation (optional)

pysodmetrics == 1.3.0
```

The package ``torch-dct`` is used for the differential discrete cosine transformation in PyTorch, and more details can be found in this [repo](https://github.com/zh217/torch-dct). **Note**: A higher version for PyTorch has included this function and it may cause some problem. You should modify the source code of ``torch-dct`` or our code to solve the problem.

The package ``pysodmetrics`` is used for calculating the metrics for camouflaged object detection based on Python, as COD and SOD share similar metrics. The usage of this package can be found in [link](https://github.com/lartpang/PySODMetrics).

***

## Data

Before testing the network, please download the data:

* CAMO dataset

```
---- CAMO
   |---- Images
       |---- Test
           |---- ****.jpg
       |---- Train
           |---- ****.jpg
   |---- GT
       |---- ****.png
```

* CHAMELEON_TestingDataset

```
---- CHAMELEON_TestingDataset
   |---- Image
       |---- ****.jpg
   |---- GT
       |---- ****.png
```

* COD10K-v3 dataset

```
---- COD10K-v3
   |---- Test
       |---- Image
           |---- ****.jpg
       |---- GT_Object
           |---- ****.png
   |---- Train
       |---- Image
           |---- ****.jpg
       |---- GT_Object
           |---- ****.png
```

**Recommendation**: you could extract these data and put them to the same folder (``e.g. ./COD_datasets/``). Then, the folder should contain three folders: ``CAMO/, CHAMELEON_TestingDataset/, COD10K-v3/``.

***

## Test and Evaluation

### Test

It is very simple to test the network. You can follow these steps:

1. You need to download the model weights [[Baidu Yun, qr1n]]( https://pan.baidu.com/s/1KD2znU2VUTQj9C__0zpS5w).

2. Change the output path in ``train.py`` Line. 36 to your need. We always use the name of the testing dataset.

```
36  -   save_supervision_path = os.path.join("results", "COD10K")
    +   save_supervision_path = '/output/path/'
```

3. You need to set the path to the testing dataset which you want in ``data_loader1.py``.

```
53    self.img_dir = 'G:/DataSet/COD10K-v3/Test/Image/'
      self.label_dir = 'G:/DataSet/COD10K-v3/Test/GT_Object/'
      # self.img_dir = '/CAMO/Images/Test/'
      # self.label_dir = '/CAMO/GT/'
      # self.img_dir = '/COD10K-v3/Test/Image/'
      # self.label_dir = '/COD10K-v3/Test/GT_Object/'
      # self.img_dir = '/CHAMELEON_TestingDataset/Image/'
      # self.label_dir = '/CHAMELEON_TestingDataset/GT/'
```

If you follow the data preparation above, you can simply use the existing code.

4. Run ``main.py``.

### Evaluation

You can use **Matlab** or **Python** script for evaluation.

* Python

You need to change the path of the ground-truth and the predictions in ``eval.py``. Using the python script is more simple and efficient.

```
6    # gt_path = 'G:/DataSet/CAMO/GT/'
     # gt_path = 'G:/DataSet/CHAMELEON_TestingDataset/GT/'
     gt_path = 'G:/DataSet/COD10K-v3/Test/GT_Object/'
     predict_path = './results/COD10K/'
```

Then run ``python eval.py``. You can get the *Smeasure, mean Emeasure, weighted Fmeasure, and MAE*.

**Note**: We also upload the results [[Baidu Yun, 1mlw]](https://pan.baidu.com/s/1r2wLfAWLww55p8gR3i7_bw).

* Matlab

You can also use the one-key evaluation toolbox for benchmarking provided by [Matlab version](https://github.com/DengPingFan/CODToolbox).

### Metrics

Finally, we get the following performance.

<table style="text-align: center">
    <tr>
        <td rowspan='2'></td>
        <td colspan='4'>COD10K</td>
        <td colspan='4'>CAMO</td>
        <td colspan='4'>CHAMELEON</td>
    </tr>
    <tr>
        <td>Sm</td>
        <td>Em</td>
        <td>wFm</td>
        <td>MAE</td>
        <td>Sm</td>
        <td>Em</td>
        <td>wFm</td>
        <td>MAE</td>
        <td>Sm</td>
        <td>Em</td>
        <td>wFm</td>
        <td>MAE</td>
    </tr>
    <tr>
        <td>Paper</td>
        <td>0.837</td>
        <td>0.918</td>
        <td>0.731</td>
        <td>0.030</td>
        <td>0.844</td>
        <td>0.898</td>
        <td>0.778</td>
        <td>0.062</td>
        <td>0.898</td>
        <td>0.949</td>
        <td>0.837</td>
        <td>0.027</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>0.8404</td>
        <td>0.9187</td>
        <td>0.7288</td>
        <td>0.0297</td>
        <td>0.8435</td>
        <td>0.8949</td>
        <td>0.7746</td>
        <td>0.0629</td>        
        <td>0.8974</td>
        <td>0.9497</td>
        <td>0.8350</td>
        <td>0.0270</td>
    </tr>
</table>

***

## Citations

If you are using this repo in a publication, please consider citing the origin paper:

```
@InProceedings{Zhong_2022_CVPR,
    author    = {Zhong, Yijie and Li, Bo and Tang, Lv and Kuang, Senyun and Wu, Shuang and Ding, Shouhong},
    title     = {Detecting Camouflaged Object in Frequency Domain},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4504-4513}
}
```

***

## Acknowledgement

* [CODToolbox](https://github.com/DengPingFan/CODToolbox)
* [PySODMetrics](https://github.com/lartpang/PySODMetrics)
* http://dpfan.net/Camouflage/
