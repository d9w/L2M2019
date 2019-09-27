from pyCGP.cgpes import CGPES
from pyCGP.cgp import CGP
from pyCGP.cgpfunctions import *
from pyCGP.evaluator import Evaluator

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
    e = L2MEvaluator(it_max=1000, ep_max=1)
    cgpFather = cgp.random(num_inputs=41, num_outputs=22, num_cols=100, num_rows=1, library=library, recurrency_distance=1.0, recursive=False)
    es = CGPES(num_offsprings=5, mutation_rate_nodes=0.1, mutation_rate_outputs=0.3, father=cgpFather, evaluator=e, folder='evo_test', num_cpus=1)
    es.run(num_iteration=50)