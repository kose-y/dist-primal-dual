# dist_pd

This repository contains the Python package `dist_pd` for distributed computation of primal-dual problems in multiple GPUs with optimal acceleration. Based on Tensorflow, we implemented distributed dense and sparse matrices, basic operations on them, and implementations of primal-dual algorithms based on these operations.   


## Installation

### On Linux

Prerequisites: Python 3, with tensorflow >=1.2 installed. The code is tested on Python 3.5.  **Tensorflow is not automatically installed by our setup script**. To install tensorflow on your system, you can follow the instructions in [this link](https://www.tensorflow.org/install/). 
If you wish to run the code on Nvidia GPU(s), please read instructions on additional setup [here](https://www.tensorflow.org/install/gpu).

One may run the following to install the code:

```bash
git clone https://github.com/kose-y/dist_pd.git
cd dist_pd
python setup.py install
```

### On Docker (recommended)

The docker version contains an option to include bigger dataset, which are shown to be scalable on multiple GPU devices.



## Running the examples

You can run the expreriments in our paper using the code in `examples/`. Read `examples/README.md` for more information.


### Obtaining the data


The "small" datasets can currently be obtained from Dropbox:

```
https://www.dropbox.com/s/d7tpa8insoq844g/ogrp_100_100_10_5000_X.mat
https://www.dropbox.com/s/t14b7p4dq66up9k/ogrp_100_100_10_5000.mat


https://www.dropbox.com/s/jc96pvzq4lo58ot/Zhu_1000_10_5000_20_0.7_100_X.mat
https://www.dropbox.com/s/tl8ode7ny8elole/Zhu_1000_10_5000_20_0.7_100.mat
```

If you are on linux, you can download the files using the script `download_data.py`. 

You may choose to download a bigger dataset from Docker Hub.

## Citation

If you use our package in your research, please cite the following item:

    Ko S, Yu D, Won J (2018). Easily parallelizable and distributable class of algorithms for structured sparsity, with optimal acceleration. arXiv preprint arXiv:1702.06234 (info on the Journal of Computational and Graphical Statistics?).
