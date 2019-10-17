import os
import sys

from pycgp.cgpes import CGPES
from pycgp.cgp import CGP
from pycgp.cgpfunctions import *
from pycgp.evaluator import Evaluator

import numpy as np

from l2mevaluator import L2MEvaluator

def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2),
            CGP.CGPFunc(f_aminus, 'aminus', 2),
            CGP.CGPFunc(f_mult, 'mult', 2),
            CGP.CGPFunc(f_exp, 'exp', 2),
            CGP.CGPFunc(f_abs, 'abs', 1),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1),
            CGP.CGPFunc(f_sqrtxy, 'sqrtxy', 2),
            CGP.CGPFunc(f_squared, 'squared', 1),
            CGP.CGPFunc(f_pow, 'pow', 2),
            CGP.CGPFunc(f_one, 'one', 0),
            CGP.CGPFunc(f_zero, 'zero', 0),
            CGP.CGPFunc(f_inv, 'inv', 1),
            CGP.CGPFunc(f_gt, 'gt', 2),
            CGP.CGPFunc(f_asin, 'asin', 1),
            CGP.CGPFunc(f_acos, 'acos', 1),
            CGP.CGPFunc(f_atan, 'atan', 1),
            CGP.CGPFunc(f_min, 'min', 2),
            CGP.CGPFunc(f_max, 'max', 2),
            CGP.CGPFunc(f_round, 'round', 1),
            CGP.CGPFunc(f_floor, 'floor', 1),
            CGP.CGPFunc(f_ceil, 'ceil', 1)
            ]

if __name__ == '__main__':
    library = build_funcLib()
    e = L2MEvaluator(it_max=2500, ep_max=1)
    cgpFather = CGP.random(num_inputs=e.num_inputs, num_outputs=e.num_outputs, num_cols=100, num_rows=1, library=library, recurrency_distance=1.0, recursive=False)
    es = CGPES(num_offsprings=5, mutation_rate_nodes=0.1, mutation_rate_outputs=0.3, father=cgpFather, evaluator=e, folder=sys.argv[1], num_cpus=1)
    es.run(num_iteration=100000)
