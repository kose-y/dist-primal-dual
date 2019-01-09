import argparse

def parser_distributed(desc, default_nslices, default_data, default_L, default_iters=10000, default_interval=1, stoc=False, default_s=None, default_output_prefix=None):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--data', dest='data_prefix', default=default_data, help='prefix for the data file, default: {}'.format(default_data))
    parser.add_argument('--L', dest='L', default=default_L, type=float, help="estimate for the Lipschitz constant of the gradient of the smooth function f. spectral norm of the data squared. default: value for our sample data.")
    #parser.add_argument('--ngroups', dest='ngrps', default=default_ngrps, type=int, help="number of groups. defaults to {}. Values should be in: {}".format(default_ngrps, ngrp_list))
    parser.add_argument('--nslices', dest='nslices', default=default_nslices, type=int, help="number of slices per GPU, intended to decrease memory usage of GPU. Default: {}".format(default_nslices))
    parser.add_argument('--ngpus', dest='ngpus', default=1, type=int, help="number of GPUs to be used. Default: 1")
    parser.add_argument('--iters', dest='iters', default=default_iters, type=int, help="number of iterations. Default: {}".format(default_iters))
    parser.add_argument('--interval', dest='interval', default=default_interval, type=int, help="interval between computing objective values. Default: {}".format(default_interval))
    parser.add_argument('--cpu', dest='use_cpu', default=False, action='store_true',
                    help="if set, CPU is used instead of GPU")
    parser.add_argument('--nonergodic', dest='nonergodic', default=False, action='store_true', help="if set, nonergodic sequence is evaluated.")
    if default_output_prefix:
        parser.add_argument('--output-prefix', dest='output_prefix', default=default_output_prefix, help="prefix for the output files. default:{}".format(default_output_prefix))
    if stoc:
        parser.add_argument('--s', dest='s', default=default_s, type=float, help="value for 's' for stochastic iterations. default: {}".format(default_s))
    args = parser.parse_args()

    return args

def parser_singledevice(desc):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--iters', dest='iters', default=10000, type=int, help="number of iterations. Default: 10000")
    parser.add_argument('--interval', dest='interval', default=1, type=int, help="interval between computing objective values. Default: 1")
    parser.add_argument('--cpu', dest='use_cpu', default=False, action='store_true',
                    help="if set, CPU is used instead of GPU")
    return parser
