# Examples

The script files in this directory reproduces the results in our paper.


Overlapping group lasso experiments:
- `group_lasso_distributed.py`: multi-GPU computation
- `group_lasso_fbf.py`: forward-backward-forward splitting
- `group_lasso_optimal.py`: optimal-rate iterations
- `group_lasso_run.py`: "base" forward-backward splitting
- `group_lasso_stoc.py`: optimal-rate stochastic iterations

Graph-guided fused lasso experiments:
- `graph_fused_distributed.py` 
- `graph_fused_fbf.py`
- `graph_fused_optimal.py`
- `graph_fused_run.py`
- `graph_fused_stoc.py`

Latent group lasso experiments:
- `latent_group_lasso_fbf.py`
- `latent_group_lasso_optimal.py`
- `latent_group_lasso_run.py`

Except for `*_distributed.py` files, where we used multiple settings, the default setting is the setting we conducted the experiments (10000 iterations, interval 1). Their outputs are in the form of MATLAB `.mat` file. If your system does not have GPU properly configured, you can run our code in CPU using the flag `--cpu`. You can change some of the settings from the command line arguments: use the flag `--help` for details. 

Optional arguments common for all files:

- `-h`, `--help`: Prints help message.
- `--iters`: Number of iterations. For optimal-rate iterations, this also defines the horizon, thus the set of parameters. 
- `--interval`: The interval between computing objecive values. Note that computing objective values is usually one of the slowest operations in many optimization problems. 
- `--cpu`: Perform the computation on CPU only. 

The following arguments are available for distributed experiments:
- `--ngroups`: Number of "groups" as defined in each experiment (graph-guided fused lasso or overlapping group lasso)
- `--nslices`: Number of slices of data to be stored in each GPU. This was introduced due to specific internal design of tensorflow in order to decrease memory usage. The default value of 5 was used in our experiments.
- `--ngpus`: Number of GPUs to be used.



For example, running

```
python group_lasso_optimal.py --interval 100 
```
runs the overlapping group lasso experiment, with the objective function computed every 100 iterations. 
