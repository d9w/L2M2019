import opensim as osim
from osim.http.client import Client
import numpy as np
import os
import sys
import argparse

from pycgp.cgpes import CGPES
from pycgp.cgp import CGP
from pycgp.evaluator import Evaluator
from l2mevaluator import L2MEvaluator
from functions import build_funcLib
from osim.env import L2M2019Env

parser = argparse.ArgumentParser(description='Evaluate or submit an individual')
parser.add_argument('--token', type=str, default='token.txt'
                    help='aicrowd token')
parser.add_argument('--ind', type=str,
                    help='CGP individual filename')
parser.add_argument('--live', action='store_true', default=False)
parser.add_argument('--visual', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for evaluation')

args = parser.parse_args()
# Settings
remote_base = 'http://osim-rl-grader.aicrowd.com/'
cgp_id = args.ind

# Create environment
if args.live:
    with open(args.token, 'r') as f:
        aicrowd_token = f.read().strip()
    client = Client(remote_base)
    observation = client.env_create(aicrowd_token, env_id='L2M2019Env')
else:
    env = L2M2019Env(visualize=args.visual)
    observation = env.reset(seed=args.seed)

# CGP controller
library = build_funcLib()
ind = CGP.load_from_file(cgp_id, library)
l2meval = L2MEvaluator(1e8, 1)
i = 0
j = 0
r_total = 0.0

while True:
    inputs = l2meval.get_inputs(observation)
    outputs = l2meval.scale_outputs(ind.run(inputs))

    if args.live:
        [observation, reward, done, info] = client.env_step(outputs.tolist())
    else:
        [observation, reward, done, info] = env.step(outputs)
    r_total += reward
    print('%d %d %f %f' % (i, j, reward, r_total))
    i += 1
    if done:
        if args.live:
            i = 0
            j += 1
            r_total = 0
            observation = client.env_reset()
            if not observation:
                break
        else:
            break

if args.live:
    client.submit()
