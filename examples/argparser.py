import argparse

def parser_distributed(desc, default_ngrps, default_nslices, ngrp_list):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--ngroups', dest='ngrps', default=default_ngrps, type=int, help="number of groups. defaults to {}. Values should be in: {}".format(default_ngrps, ngrp_list))
    parser.add_argument('--nslices', dest='nslices', default=5, type=int, help="number of slices per GPU, intended to decrease memory usage of GPU. Default: {}".format(default_nslices))
    parser.add_argument('--ngpus', dest='ngpus', default=1, type=int, help="number of GPUs to be used. Default: 1")
    parser.add_argument('--iters', dest='iters', default=1100, type=int, help="number of iterations. Default: 1100")
    parser.add_argument('--interval', dest='interval', default=100, type=int, help="interval between computing objective values. Default: 100")
    parser.add_argument('--cpu', dest='use_cpu', default=False, action='store_true',
                    help="if set, CPU is used instead of GPU")
    return parser

def parser_singledevice(desc):
    parser = argparse.ArgumentParser(desc)
    parser.add_argument('--iters', dest='iters', default=10000, type=int, help="number of iterations. Default: 10000")
    parser.add_argument('--interval', dest='interval', default=1, type=int, help="interval between computing objective values. Default: 1")
    parser.add_argument('--cpu', dest='use_cpu', default=False, action='store_true',
                    help="if set, CPU is used instead of GPU")
    return parser
