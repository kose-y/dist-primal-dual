# dist_pd

## Installation

### On Linux

Prerequisites: Python 3, with tensorflow >=1.2 installed. The code is tested on Python 3.6.  **Tensorflow is not automatically installed by our setup script**.

One may run the following to install the code:

```bash
git clone https://github.com/kose-y/dist_pd.git
cd dist_pd
python setup.py --install
```

### On Docker (recommended)

The docker version comes with bigger dataset, which are shown to be scalable on multiple GPU devices.

First, obtain docker on your system. 

Then ...




## Running the examples

## Obtaining the data

The Docker version comes with some of the bigger datasets.

The "small" datasets can currently be obtained from Dropbox:

```
https://www.dropbox.com/s/d7tpa8insoq844g/ogrp_100_100_10_5000_X.mat
https://www.dropbox.com/s/t14b7p4dq66up9k/ogrp_100_100_10_5000.mat


https://www.dropbox.com/s/jc96pvzq4lo58ot/Zhu_1000_10_5000_20_0.7_100_X.mat
https://www.dropbox.com/s/tl8ode7ny8elole/Zhu_1000_10_5000_20_0.7_100.mat
```

If you are on linux, you can download the files using the script `download_data.py`. 

