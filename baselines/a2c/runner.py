import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, q_exp, q_model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        nenv = self.nenv
        self.batch_ob_shape_acer = (nenv*(nsteps+1),) + env.observation_space.shape
        #print("!!!!!! a2c batch_action_shape: " + str(model.train_model.action.shape.as_list()))
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        #print("a2c batch action shape: " + str(self.batch_action_shape))
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape_acer[-1] // self.nstack
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.q_exp = q_exp
        self.q_model = q_model
        self.lam=0.95

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_obs_acer, mb_rewards, mb_actions, mb_values, mb_dones, mb_dones_ppo2, mb_mus, mb_neglogpacs = [], [], [], [], [],[],[],[],[]
        mb_states = self.states
        epinfos = []
        #print("A2C self.obs: ")
        #print(np.shape(self.obs))
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            _, mus, _ = self.model._step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_obs_acer.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_dones_ppo2.append(self.dones)
            mb_neglogpacs.append(neglogpacs)

            # Take actions in env and look the results
            #print("A2C actions zise: ")
            #print(np.shape(actions))
            obs, rewards, dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
        mb_obs_acer.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_acer = np.asarray(mb_obs_acer, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_ppo2 = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards_acer = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards_ppo2 = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions_acer = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_actions_ppo2 = np.asarray(mb_actions)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values_ppo2 = np.asarray(mb_values, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones_ppo2 = np.asarray(mb_dones_ppo2, dtype=np.bool)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks_acer = mb_dones
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        #print("a2c self.obs shape at last_values: ")
        #print(np.shape(self.obs))

        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        #print("a2c last_values shape: ")
        #print(np.shape(last_values))
        #print(last_values)


        #print("a2c mb_actions_a2c size: ")
        #print(np.shape(mb_actions))
        #print("a2c mb_actions_acer size: ")
        #print(np.shape(mb_actions_acer))
        #print("a2c mb_actions_ppo2 size: ")
        #print(np.shape(mb_actions_ppo2))

        mb_returns = np.zeros_like(mb_rewards_ppo2)
        mb_advs = np.zeros_like(mb_rewards_ppo2)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones_ppo2[t+1]
                nextvalues = mb_values_ppo2[t+1]
            delta = mb_rewards_ppo2[t] + self.gamma * nextvalues * nextnonterminal - mb_values_ppo2[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values_ppo2

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_value = last_values.tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_value)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()


        exp_acer = [enc_obs, mb_obs_acer, mb_actions_acer, mb_rewards_acer, mb_mus, mb_dones, mb_masks_acer]

        ll = list(map(sf01, (mb_obs_ppo2, mb_returns, mb_dones_ppo2, mb_actions_ppo2, mb_values_ppo2, mb_neglogpacs)))
        exp_ppo2 = [ll[0], ll[1], ll[2], ll[3], ll[4], ll[5], mb_states, epinfos]

        self.q_exp[1].put(exp_ppo2)
        self.q_exp[2].put(exp_acer)

        ret = []
        ret.append([mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos])

        while not self.q_exp[0].empty():
            exp_a2c = self.q_exp[0].get()
            ret.append(exp_a2c)

        #return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos
        return ret

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
