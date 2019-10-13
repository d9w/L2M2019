import opensim as osim
from osim.http.client import Client
import numpy as np
import os
import sys

from pycgp.cgpes import CGPES
from pycgp.cgp import CGP
from pycgp.evaluator import Evaluator
from l2mevaluator import L2MEvaluator
from functions import build_funcLib
from osim.env import L2M2019Env

# Settings
remote_base = "http://osim-rl-grader.aicrowd.com/"
aicrowd_token = "f329c983f0be21ed880efab83d09f692" # use your aicrowd token
cgp_id = 'output/cgp_genome_1_13.810665266753004.txt'

# Create environment
#client = Client(remote_base)
#observation = client.env_create(aicrowd_token, env_id='L2M2019Env')
env = L2M2019Env(visualize=False)
observation = env.reset()

# CGP controller
library = build_funcLib()
ind = CGP.load_from_file(cgp_id, library)
l2meval = L2MEvaluator(1e8, 1)
i = 0
j = 0

while True:
    #action = my_controller.update(observation)
    inputs = l2meval.get_inputs(observation)
    outputs = l2meval.scale_outputs(ind.run(inputs))

    #[observation, reward, done, info] = client.env_step(outputs)
    [observation, reward, done, info] = env.step(outputs)
    print('%d %d %f' % (i, j, reward))
    i += 1
    if done:
        break
        i = 0
        j += 1
        # observation = client.env_reset()
        if not observation:
            break

# client.submit()
