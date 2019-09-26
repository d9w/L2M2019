from pyCGP.cgpes import CGPES
from pyCGP.cgp import CGP
from pyCGP.cgpfunctions import *
from pyCGP.evaluator import Evaluator

from osim.env import L2M2019Env
import numpy as np
from utils import *

class L2MEvaluator(Evaluator):
    def __init__(self, it_max, ep_max):
        super().__init__()
        self.it_max = it_max
        self.ep_max = ep_max
        self.env = L2M2019Env(visualize=False)

    def evaluate(self, cgp, it):
        np.random.seed(it)
        for e in range(self.ep_max):
            # resetting env
            obs = self.env.reset()

            done = False
            fit = 0
            while not done:
                # parsing inputs
                inputs = np.zeros(41)
                inputs[0] =  change_interval(obs['pelvis']['height'], self.env.observation_space.low[242], self.env.observation_space.high[242], -1, 1)
                inputs[1] =  change_interval(obs['pelvis']['pitch'], self.env.observation_space.low[243], self.env.observation_space.high[243], -1, 1)
                inputs[2] =  change_interval(obs['pelvis']['roll'], self.env.observation_space.low[244], self.env.observation_space.high[244], -1, 1)
                inputs[3] =  change_interval(obs['pelvis']['vel'][0], self.env.observation_space.low[245], self.env.observation_space.high[245], -1, 1)
                inputs[4] =  change_interval(obs['pelvis']['vel'][1], self.env.observation_space.low[246], self.env.observation_space.high[246], -1, 1)
                inputs[5] =  change_interval(obs['pelvis']['vel'][2], self.env.observation_space.low[247], self.env.observation_space.high[247], -1, 1)
                inputs[6] =  change_interval(obs['pelvis']['vel'][3], self.env.observation_space.low[248], self.env.observation_space.high[248], -1, 1)
                inputs[7] =  change_interval(obs['pelvis']['vel'][4], self.env.observation_space.low[249], self.env.observation_space.high[249], -1, 1)
                inputs[8] =  change_interval(obs['pelvis']['vel'][5], self.env.observation_space.low[250], self.env.observation_space.high[250], -1, 1)

                inputs[9] =  change_interval(obs['r_leg']['ground_reaction_forces'][2], self.env.observation_space.low[253], self.env.observation_space.high[253], -1, 1)
                inputs[10] =  change_interval(obs['r_leg']['joint']['hip_abd'], self.env.observation_space.low[254], self.env.observation_space.high[254], -1, 1)
                inputs[11] =  change_interval(obs['r_leg']['joint']['hip'], self.env.observation_space.low[255], self.env.observation_space.high[255], -1, 1)
                inputs[12] =  change_interval(obs['r_leg']['joint']['knee'], self.env.observation_space.low[256], self.env.observation_space.high[256], -1, 1)
                inputs[13] =  change_interval(obs['r_leg']['joint']['ankle'], self.env.observation_space.low[257], self.env.observation_space.high[257], -1, 1)
                inputs[14] =  change_interval(obs['r_leg']['HAB']['f'], self.env.observation_space.low[262], self.env.observation_space.high[262], -1, 1)
                inputs[15] =  change_interval(obs['r_leg']['HAD']['f'], self.env.observation_space.low[265], self.env.observation_space.high[265], -1, 1)
                inputs[16] =  change_interval(obs['r_leg']['HFL']['f'], self.env.observation_space.low[268], self.env.observation_space.high[268], -1, 1)
                inputs[17] =  change_interval(obs['r_leg']['GLU']['f'], self.env.observation_space.low[271], self.env.observation_space.high[271], -1, 1)
                inputs[18] =  change_interval(obs['r_leg']['HAM']['f'], self.env.observation_space.low[274], self.env.observation_space.high[274], -1, 1)
                inputs[19] =  change_interval(obs['r_leg']['RF']['f'], self.env.observation_space.low[277], self.env.observation_space.high[277], -1, 1)
                inputs[20] =  change_interval(obs['r_leg']['VAS']['f'], self.env.observation_space.low[280], self.env.observation_space.high[280], -1, 1)
                inputs[21] =  change_interval(obs['r_leg']['BFSH']['f'], self.env.observation_space.low[283], self.env.observation_space.high[283], -1, 1)
                inputs[22] =  change_interval(obs['r_leg']['GAS']['f'], self.env.observation_space.low[286], self.env.observation_space.high[286], -1, 1)
                inputs[23] =  change_interval(obs['r_leg']['SOL']['f'], self.env.observation_space.low[289], self.env.observation_space.high[289], -1, 1)
                inputs[24] =  change_interval(obs['r_leg']['TA']['f'], self.env.observation_space.low[292], self.env.observation_space.high[292], -1, 1)

                inputs[25] =  change_interval(obs['l_leg']['ground_reaction_forces'][2], self.env.observation_space.low[297], self.env.observation_space.high[297], -1, 1)
                inputs[26] =  change_interval(obs['l_leg']['joint']['hip_abd'], self.env.observation_space.low[298], self.env.observation_space.high[298], -1, 1)
                inputs[27] =  change_interval(obs['l_leg']['joint']['hip'], self.env.observation_space.low[299], self.env.observation_space.high[299], -1, 1)
                inputs[28] =  change_interval(obs['l_leg']['joint']['knee'], self.env.observation_space.low[300], self.env.observation_space.high[300], -1, 1)
                inputs[29] =  change_interval(obs['l_leg']['joint']['ankle'], self.env.observation_space.low[301], self.env.observation_space.high[301], -1, 1)
                inputs[30] =  change_interval(obs['l_leg']['HAB']['f'], self.env.observation_space.low[306], self.env.observation_space.high[306], -1, 1)
                inputs[31] =  change_interval(obs['l_leg']['HAD']['f'], self.env.observation_space.low[309], self.env.observation_space.high[309], -1, 1)
                inputs[32] =  change_interval(obs['l_leg']['HFL']['f'], self.env.observation_space.low[312], self.env.observation_space.high[312], -1, 1)
                inputs[33] =  change_interval(obs['l_leg']['GLU']['f'], self.env.observation_space.low[315], self.env.observation_space.high[315], -1, 1)
                inputs[34] =  change_interval(obs['l_leg']['HAM']['f'], self.env.observation_space.low[318], self.env.observation_space.high[318], -1, 1)
                inputs[35] =  change_interval(obs['l_leg']['RF']['f'], self.env.observation_space.low[321], self.env.observation_space.high[321], -1, 1)
                inputs[36] =  change_interval(obs['l_leg']['VAS']['f'], self.env.observation_space.low[324], self.env.observation_space.high[324], -1, 1)
                inputs[37] =  change_interval(obs['l_leg']['BFSH']['f'], self.env.observation_space.low[327], self.env.observation_space.high[327], -1, 1)
                inputs[38] =  change_interval(obs['l_leg']['GAS']['f'], self.env.observation_space.low[330], self.env.observation_space.high[330], -1, 1)
                inputs[39] =  change_interval(obs['l_leg']['SOL']['f'], self.env.observation_space.low[333], self.env.observation_space.high[333], -1, 1)
                inputs[40] =  change_interval(obs['l_leg']['TA']['f'], self.env.observation_space.low[336], self.env.observation_space.high[336], -1, 1)


                pelvis = list(flatten(obs["pelvis"].values()))

                r_leg = obs["r_leg"]["ground_reaction_forces"]
                r_leg += list(obs["r_leg"]["joint"].values())
                for x in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                    r_leg += [obs['r_leg'][x]['f']]

                l_leg = obs["l_leg"]["ground_reaction_forces"]
                l_leg += list(obs["l_leg"]["joint"].values())
                for x in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                    l_leg += [obs['l_leg'][x]['f']]

                inputs = pelvis + r_leg + l_leg

                print(inputs)


                #building inputs


                obs, reward, done, info = self.env.step(self.env.action_space.sample())


