# dist-primal-dual

This repository contains the Python package `dist_pd` for distributed computation of primal-dual problems in multiple GPUs with optimal acceleration, published in

* S. Ko, D. Yu, J. Won. Easily Parallelizable and Distributable Class of Algorithms for Structured Sparsity, With Optimal Acceleration. [J. Comput. Graph. Stat.](https://doi.org/10.1080/10618600.2019.1592757) 2019:28(4), 821- 2833.

Based on Tensorflow, we implemented distributed dense and sparse matrices, basic operations on them, and implementations of primal-dual algorithms based on these operations.   


## Installation


### On Docker (recommended)

We have all the prerequisites readily installed in our docker containers. See `docker/`. 
The docker version contains an option to also download bigger datasets, which is shown to be scalable on multiple GPU devices.

### On Linux

Prerequisites: Python 3, with tensorflow >=1.2 installed. The code is tested on Python 3.5.  **Tensorflow is not automatically installed by our setup script**. To install tensorflow on your system, you can follow the instructions in [this link](https://www.tensorflow.org/install/). 
If you wish to run the code on Nvidia GPU(s), please read instructions on additional setup [here](https://www.tensorflow.org/install/gpu).

One may run the following to install the code:

```bash
git clone https://github.com/kose-y/dist_pd.git
cd dist_pd
python setup.py install
```

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

If you are using Docker, these datasets come with the package. 
You may also choose to download a bigger dataset from Docker Hub: see `docker/README.md`.

If you are on linux, you can download the files using the script `download_data.py`. 

## Citation

If you use our package in your research, please cite the following item:

```
@article{doi:10.1080/10618600.2019.1592757,
    author = {Seyoon Ko and Donghyeon Yu and Joong-Ho Won},
    title = {Easily Parallelizable and Distributable Class of Algorithms for Structured Sparsity, with Optimal Acceleration},
    journal = {Journal of Computational and Graphical Statistics},
    volume = {28},
    number = {4},
    pages = {821-833},
    year  = {2019},
    publisher = {Taylor & Francis},
    doi = {10.1080/10618600.2019.1592757},
    URL = { 
        https://doi.org/10.1080/10618600.2019.1592757
    },
    eprint = { 
        https://doi.org/10.1080/10618600.2019.1592757
    }
}
```

* [DOI link](https://www.tandfonline.com/doi/full/10.1080/10618600.2019.1592757)
* [arXiv preprint](https://arxiv.org/abs/1702.06234)
    * Note: [Older version](https://arxiv.org/abs/1702.06234v1) of the manuscript was titled "On a Class of First-order Primal-Dual Algorithms for Composite Convex Minimization Problems".
