[![Stars](https://img.shields.io/github/stars/JunfeiYi/PSTL-Net)](
https://github.com/JunfeiYi/PSTL-Net)
[![Open issue](https://img.shields.io/github/issues/JunfeiYi/PSTL-Net)](
https://github.com/JunfeiYi/PSTL-Net/issues)
[![Source Code](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/JunfeiYi/PSTL-Net/archive/refs/heads/master.zip)


# PSTL-Net

#### English 

This is the official implemtentation of Patchwise Self-Texture-Learning Network (PSTL-Net) in the paper PSTL-Net: Patchwise Self-Texture-Learning Network for Transmission Line Inspection.
 
# Getting Started

For `PSTL-Net`, the [mmdetection](https://github.com/open-mmlab/mmdetection) is used. More installation and usage please refer to the [mmdetection](https://github.com/open-mmlab/mmdetection).


1. ## Get Source Codes
    ```bash
    git clone https://github.com/JunfeiYi/PSTL-Net.git
    cd PSTL-Net
    pip install -r requirements.txt  # install dependencies
    python setup.py develop
    ```
2. ## Prepare Datasets and Trained Weights

   Trained Weights:
   
   Baidu Disk: https://pan.baidu.com/s/108Ez-u4HdZQHsLCZOqUiog, Password: 7rrd 
    

3. ## Train model

    To train a model, run:
    ```bash
    python tools/train.py configs/0PSTL-Net/PSTL-Net.py
    ```
    After training, the model will be saved in `work_dirs/`.

4. ## Evaluate
    After training or download trained weights, you can evaluate the model by running:
    ```bash
    # evaluate the model on the testset
    python tools/analysis_tools/eval_metric.py configs/0PSTL-Net/PSTL-Net.py result.pkl --eval bbox 
    python tools/test.py configs/0PSTL-Net/PSTL-Net.py best.pth --out result.pkl  # generate result.pkl
    ```
    Detailed tutorials of the mmdetection are available in [mmdtection/get_started](https://mmdetection.readthedocs.io/en/latest/get_started.html).
 
# Contributors
The PSTL-Net is authored by [Junfei Yi](https://junfeiyi.github.io/)\*.

If you have any questions, please new an [issue](https://github.com/JunfeiYi/PSTL-Net/issues), thank you for your attention!

Please **Star** this project and **Cite** this paper if its helpful to you .

  ```bash
  @ARTICLE{10375333,
  author={Yi, Junfei and Mao, Jianxu and Zhang, Hui and Zeng, Kai and Tao, Ziming and Zhong, Hang and Wang, Shaoyuan and Wang, Yaonan},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={PSTL-Net: A Patchwise Self-Texture-Learning Network for Transmission Line Inspection}, 
  year={2024},
  volume={73},
  number={},
  pages={1-14},
  doi={10.1109/TIM.2023.3341118}}
  ```


# License
THe PSTL-Net is freely available for non-commercial use, and may be redistributed under these conditions. 
For commercial queries, please drop an e-mail to yijunfei@hnu.edu.cn, we will send the detail agreement to you.


