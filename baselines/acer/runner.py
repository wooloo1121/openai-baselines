import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces
from baselines.a2c.utils import discount_with_dones


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, q_exp, q_model):
        super().__init__(env=env, model=model, nsteps=nsteps)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape_acer = (nenv*(nsteps+1),) + env.observation_space.shape
        self.batch_shape = (4*(2048+1),) + env.observation_space.shape

        self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape_acer[-1] // self.nstack
        self.q_exp = q_exp
        self.q_model = q_model
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        #print("!!!!! acer batch_action_shape: " + str(model.train_model.action.shape.as_list()))
        self.batch_action_shape = [2048]
        #[x if x is not None else -1 for x in model.train_model.action.shape.as_list()]

        self.gamma = 0.99
        self.lam=0.95

    def run(self):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_obs_acer, mb_actions, mb_mus, mb_dones, mb_dones_ppo2, mb_rewards, mb_values, mb_neglogpacs = [], [], [], [], [], [], [], [], []
        #mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            #print("ACER self.obs: ")
            #print(np.shape(self.obs))
            actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones)
            _, values, _, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            #print("acer step values: " + str(values))
            mb_obs.append(np.copy(self.obs))
            mb_obs_acer.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_dones_ppo2.append(self.dones)
            tmp = np.zeros(4)
            for i in range(4):
                for j in range(6):
                    tmp[i] += values[i,j] * mus[i,j]
            values = tmp
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            #print("ACER actions size: ")
            #print(np.shape(actions))
            obs, rewards, dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
        mb_obs_acer.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_acer = np.asarray(mb_obs_acer, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_ppo2 = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_actions_acer = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_actions_ppo2 = np.asarray(mb_actions)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_rewards_acer = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards_ppo2 = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_values_ppo2 = np.asarray(mb_values, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)

        mb_dones_ppo2 = np.asarray(mb_dones_ppo2, dtype=np.bool)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks_acer = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        #print("acer mb_actions_a2c size: ")
        #print(np.shape(mb_actions))
        #print("acer mb_actions_acer size: ")
        #print(np.shape(mb_actions_acer))
        #print("acer mb_actions_ppo2 size: ")
        #print(np.shape(mb_actions_ppo2))

        #print("acer self.obs shape at last_values: ")
        #print(np.shape(self.obs))

        #print("acer self.states shape: ")
        #print(np.shape(self.states))
        #print("acer self.dones shape: ")
        #print(np.shape(self.dones))
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        #print("acer last_values shape: ")
        #print(np.shape(last_values))
        #print(last_values)
        _, prop, _ = self.model._step(self.obs, S=self.states, M=self.dones)
        #print("acer action probability: ")
        #print(np.shape(prop))
        #print(prop)
        tmp = np.zeros(4)
        for i in range(4):
            for j in range(6):
                tmp[i] += last_values[i,j] * prop[i,j]
        last_values = tmp
        #print("acer tmp: " + str(tmp))

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

        # discount/bootstrap off value fn
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
            #print(nextvalues)
            #print(mb_rewards_ppo2[t])
            delta = mb_rewards_ppo2[t] + self.gamma * nextvalues * nextnonterminal - mb_values_ppo2[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values_ppo2

        exp_a2c = [mb_obs, None, mb_rewards.flatten(), mb_masks.flatten(), mb_actions.reshape(self.batch_action_shape), mb_values.flatten(), epinfos]

        ll = list(map(sf01, (mb_obs_ppo2, mb_returns, mb_dones_ppo2, mb_actions_ppo2, mb_values_ppo2, mb_neglogpacs)))
        exp_ppo2 = [ll[0], ll[1], ll[2], ll[3], ll[4], ll[5], None, epinfos]

        self.q_exp[0].put(exp_a2c)
        self.q_exp[1].put(exp_ppo2)

        ret = []
        #print("mb_obs_acer shape before return: ")
        #print(np.shape(mb_obs_acer))
        ret.append([enc_obs, mb_obs_acer, mb_actions_acer, mb_rewards_acer, mb_mus, mb_dones, mb_masks_acer])

        while not self.q_exp[2].empty():
            exp_acer = self.q_exp[2].get()
            ret.append(exp_acer)

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        #return enc_obs, mb_obs_acer, mb_actions_acer, mb_rewards_acer, mb_mus, mb_dones, mb_masks_acer
        return ret

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
