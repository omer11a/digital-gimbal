# Digital Gimbal
Official implementation of CVPR21 "Digital Gimbal: End-to-end Deep Image Stabilization with Learnable Exposure Times"

[\[paper\]](https://arxiv.org/abs/2012.04515)

<img src="https://user-images.githubusercontent.com/33555542/117542573-db06d700-b021-11eb-84f5-a42dcbaa2dbc.png" width="500"/>

## Instructions

### Installing requirements

```
pip3 install -r requirements.txt
```

### Training

```
python3 run.py -nw <num_cpu_workers> -c <config_file> -r
```

Drop ```-r``` if you resume training from a previous checkpoint.

The configuration file determines the settings of the model, as well as the locations of the training/test datasets and checkpoint directory.

We supply ```config.ini```, which is the main configuration file used for producing the synthetic experiments in the paper.

### Inference

```
python3 run.py -nw <num_cpu_workers> -c <config_file> -e
```

The configuration file used here must match the one used during training, with a valid checkpoint located in the ceckpoint directory.

To reproduce our synthetic experiments, download our pre-trained model using this [link](https://drive.google.com/file/d/1x4m2nVveAKXTIcmlT2zkxhzigO8NcOpD/view?usp=sharing), place the checkpoint file inside the checkpoint directory and use ```config.ini``` as your configuration file.
