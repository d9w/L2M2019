from pycgp.cgpes import CGPES
from pycgp.cgp import CGP
from pycgp.cgpfunctions import *
from pycgp.evaluator import Evaluator

from osim.env import L2M2019Env
import numpy as np
from pycgp.utils import *

class L2MEvaluator(Evaluator):
    num_inputs = 43
    num_outputs = 22

    def __init__(self, it_max, ep_max):
        super().__init__()
        self.it_max = it_max
        self.ep_max = ep_max
        self.env = L2M2019Env(visualize=False, difficulty=3)
#        self.obs_high = np.array(self.env.observation_space.high)
#        self.obs_low = np.array(self.env.observation_space.low)
        self.stop_measure = 0
        self.patience = 5

    def evaluate(self, cgp, generation):
        np.random.seed(generation)
        fit = 0
        for e in range(self.ep_max):
            # resetting env
            obs = self.env.reset()
            inputs = self.get_inputs(obs)
            self.stop_measure = inputs[0]
            done = False
            it = 0
            patience_count = 0
            while (not done) and (it < self.it_max):
                # parsing inputs
                inputs = self.get_inputs(obs)
                outputs = self.scale_outputs(cgp.run(inputs))
                obs, reward, done, info = self.env.step(outputs)
                fit += reward
                it += 1
                if reward == 0.1:
                    patience_count += 1
                else:
                    patience_count = 0
                if np.abs(inputs[0] - self.stop_measure) > (0.25 * self.stop_measure):
                    break
                if patience_count > self.patience:
                    break
        return fit

    def get_normalized_obs(self):
        obs = np.array(self.env.get_observation_clipped())
        return 2.0 * ((obs - self.obs_low) / (self.obs_high - self.obs_low)) - 1.0

    def get_inputs(self, obs):
        inputs = np.zeros(self.num_inputs)
        inputs[0] =  change_interval(obs['pelvis']['height'], self.env.observation_space.low[242], self.env.observation_space.high[242], 0, 1)
        inputs[1] =  change_interval(obs['pelvis']['pitch'], self.env.observation_space.low[243], self.env.observation_space.high[243], -1, 1)
        inputs[2] =  change_interval(obs['pelvis']['roll'], self.env.observation_space.low[244], self.env.observation_space.high[244], -1, 1)
        inputs[3] =  change_interval(obs['pelvis']['vel'][0], self.env.observation_space.low[245], self.env.observation_space.high[245], -1, 1)
        inputs[4] =  change_interval(obs['pelvis']['vel'][1], self.env.observation_space.low[246], self.env.observation_space.high[246], -1, 1)
        inputs[5] =  change_interval(obs['pelvis']['vel'][2], self.env.observation_space.low[247], self.env.observation_space.high[247], -1, 1)
        inputs[6] =  change_interval(obs['pelvis']['vel'][3], self.env.observation_space.low[248], self.env.observation_space.high[248], -1, 1)
        inputs[7] =  change_interval(obs['pelvis']['vel'][4], self.env.observation_space.low[249], self.env.observation_space.high[249], -1, 1)
        inputs[8] =  change_interval(obs['pelvis']['vel'][5], self.env.observation_space.low[250], self.env.observation_space.high[250], -1, 1)

        inputs[9] =  change_interval(obs['r_leg']['ground_reaction_forces'][2], self.env.observation_space.low[253], self.env.observation_space.high[253], -1, 1)
        # TODO: bounds not well defined
        inputs[10] =  change_interval(obs['r_leg']['joint']['hip_abd'], self.env.observation_space.low[254], self.env.observation_space.high[254], -1, 1)
        inputs[11] =  change_interval(obs['r_leg']['joint']['hip'], self.env.observation_space.low[255], self.env.observation_space.high[255], -1, 1)
        inputs[12] =  change_interval(obs['r_leg']['joint']['knee'], self.env.observation_space.low[256], self.env.observation_space.high[256], -1, 1)
        # TODO: end bounds not well defined
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
        # TODO: bounds not well defined
        inputs[26] =  change_interval(obs['l_leg']['joint']['hip_abd'], self.env.observation_space.low[298], self.env.observation_space.high[298], -1, 1)
        inputs[27] =  change_interval(obs['l_leg']['joint']['hip'], self.env.observation_space.low[299], self.env.observation_space.high[299], -1, 1)
        inputs[28] =  change_interval(obs['l_leg']['joint']['knee'], self.env.observation_space.low[300], self.env.observation_space.high[300], -1, 1)
        # TODO: end bounds not well defined
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

        inputs[41] = change_interval(np.sum(np.sum(obs['v_tgt_field'][0])), 121.0 * self.env.observation_space.low[0], 121.0 * self.env.observation_space.high[0], -1, 1)
        inputs[42] = change_interval(np.sum(np.sum(obs['v_tgt_field'][1])), 121.0 * self.env.observation_space.low[0], 121.0 * self.env.observation_space.high[0], -1, 1)
        return np.clip(inputs, -1.0, 1.0)

    def scale_outputs(self, outputs):
        for i in range(len(outputs)):
            outputs[i] = change_interval(outputs[i], -1, 1, self.env.action_space.low[i], self.env.action_space.high[i])
        return outputs

    def clone(self):
        return L2MEvaluator(self.it_max, self.ep_max)
