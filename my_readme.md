# CMSC733-Project 4 | Learning Structure from Motion | An Unsuperwised way
This codebase implements the system described in the paper:

`Unsupervised Learning of Depth and Ego-Motion from Video`

## Preamble
This codebase was developed and tested with Pytorch 1.0.1, CUDA 10 and Ubuntu 16.04. Original code was developped in tensorflow, you can access it [here](https://github.com/tinghuiz/SfMLearner)

## Prerequisite

```bash
pip3 install -r requirements.txt
```

## Preparing training data
Preparation is roughly the same command as in the original code.

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command. The `--with-depth` option will save resized copies of groundtruth to help you setting hyper parameters. The `--with-pose` will dump the sequence pose in the same format as Odometry dataset (see pose evaluation)
```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti_raw' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 128 --num-threads 4 [--static-frames /path/to/static_frames.txt] [--with-depth] [--with-pose]
```


For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. You will probably need to contact the administrators to be able to get it. Then run the following command
```bash
python3 data/prepare_train_data.py /path/to/cityscapes/dataset/ --dataset-format 'cityscapes' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 171 --num-threads 4
```
Notice that for Cityscapes the `img_height` is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python3 train.py /path/to/the/formatted/data/ -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output [--with-gt]
```
You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~30K iterations when training on KITTI.

## Evaluation

Disparity map generation can be done with `run_inference.py`
```bash
python3 run_inference.py --pretrained /path/to/dispnet --dataset-dir /path/pictures/dir --output-dir /path/to/output/dir
```
Will run inference on all pictures inside `dataset-dir` and save a jpg of disparity (or depth) to `output-dir` for each one see script help (`-h`) for more options.

Disparity evaluation is avalaible
```bash
python3 test_disp.py --pretrained-dispnet /path/to/dispnet --pretrained-posenet /path/to/posenet --dataset-dir /path/to/KITTI_raw --dataset-list /path/to/test_files_list
```

Test file list is available in kitti eval folder. To get fair comparison with [Original paper evaluation code](https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py), don't specify a posenet. However, if you do,  it will be used to solve the scale factor ambiguity, the only ground truth used to get it will be vehicle speed which is far more acceptable for real conditions quality measurement, but you will obviously get worse results.

Pose evaluation is also available on [Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Be sure to download both color images and pose !

```bash
python3 test_pose.py /path/to/posenet --dataset-dir /path/to/KITIT_odometry --sequences [09]
```

**ATE** (*Absolute Trajectory Error*) is computed as long as **RE** for rotation (*Rotation Error*). **RE** between `R1` and `R2` is defined as the angle of `R1*R2^-1` when converted to axis/angle. It corresponds to `RE = arccos( (trace(R1 @ R2^-1) - 1) / 2)`.
While **ATE** is often said to be enough to trajectory estimation, **RE** seems important here as sequences are only `seq_length` frames long.

## Pretrained Nets
We trained a total of 5 models, first is the original model and others are our modified networks.
Download the model checkpoints from [here](https://drive.google.com/drive/folders/1Hw9DFBvIDYwRfX3ll6PSuk8s3xZPmgB5?usp=sharing)

Arguments used :

```bash
python3 train.py /path/to/the/formatted/data/ -b4 -m0 -s2.0 --epoch-size 1000 --sequence-length 5 --log-output --with-gt
```
