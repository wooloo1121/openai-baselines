import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.a2c.utils import discount_with_dones
import tensorflow as tf
import math

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, q_exp=None, q_model=None, EVAL=False, model_a2c=None, model_acer=None):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.q_exp = q_exp
        self.q_model = q_model
        self.model_a2c = model_a2c
        self.model_acer = model_acer
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        #print("!!!!! ppo2 batch_action_shape: " + str(model.train_model.action.shape.as_list()))
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.EVAL = EVAL

        nenv = self.nenv
        self.nstack = self.env.nstack
        self.batch_ob_shape_acer = (nenv*(nsteps+1),) + env.observation_space.shape
        self.nc = self.batch_ob_shape_acer[-1] // self.nstack
        self.models = [model_a2c, model, model_acer]
        self.nact = self.env.action_space.n

    def run(self):

        a2c_param = None
        while not self.q_model[0].empty():
            a2c_param = self.q_model[0].get()
        if a2c_param:
            params = tf.trainable_variables("a2c_model")
            for i in range(len(params)):
                #params[i].assign(a2c_param[i])
                update = tf.assign(params[i],a2c_param[i])
                self.models[0].sess.run(update)
        acer_param = None
        while not self.q_model[2].empty():
            acer_param = self.q_model[2].get()
        if acer_param:
            params = tf.trainable_variables("acer_model")
            for i in range(len(params)-2):
                #params[i].assign(acer_param[i])
                update = tf.assign(params[i],acer_param[i])
                self.models[2].sess.run(update)


        # Here, we init the lists that will contain the mb of experiences
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_obs_acer, mb_rewards, mb_rewards_a2c, mb_actions, mb_values, mb_values_a2c, mb_dones, mb_dones_acer, mb_neglogpacs, mb_neglogpacs_a2c, mb_neglogpacs_acer, mb_mus = [], [], [], [], [], [], [], [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        count = [0,0,0]
        for _ in range(self.nsteps):
            #print("PPO2 self.obs: ")
            #print(np.shape(self.obs))
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            action_list = []
            value_list = []
            state_list = []
            likelihood_list = []
            mus_list = []
            for k in range(3):
                tmp0, tmp1, tmp2, tmp3 = self.models[k].step(self.obs, S=self.states, M=self.dones)
                _, tmp4, _ = self.models[k]._step(self.obs, S=self.states, M=self.dones)
                action_list.append(tmp0)
                value_list.append(tmp1)
                state_list.append(tmp2)
                likelihood_list.append(tmp3)
                mus_list.append(tmp4)
            for k in range(4):
                temp = [likelihood_list[0][k],likelihood_list[1][k],likelihood_list[2][k]]
                temp_min = min(temp)
                index = 1
                threshold = -1 * math.log(1/self.nact) / 2
                if temp_min < threshold:
                    index = temp.index(temp_min)
                count[index] += 1
                action_list[0][k] = action_list[index][k]
                #print("selected action: ")
                #print(action_list[0][k])
                #value_list[0][k] = value_list[index][k]
                #state_list[0][k] = state_list[index][k]
                #likelihood_list[0][k] = likelihood_list[index][k]
                #mus_list[0][k] = mus_list[index][k]
                likelihood_list[0][k] = -1 * math.log(mus_list[0][k][action_list[0][k]])
                likelihood_list[1][k] = -1 * math.log(mus_list[1][k][action_list[0][k]])
                likelihood_list[2][k] = -1 * math.log(mus_list[2][k][action_list[0][k]])


            #actions, values, states, neglogpacs = self.models[index].step(self.obs, S=self.states, M=self.dones)
            #_, mus, _ = self.models[index]._step(self.obs, S=self.states, M=self.dones)

            actions = action_list[0]
            values = value_list[1]
            values_a2c = value_list[0]
            self.states = state_list[0]
            neglogpacs = likelihood_list[1]
            neglogpacs_a2c = likelihood_list[0]
            neglogpacs_acer = likelihood_list[2]
            mus = mus_list[1]


            #_, mus, _ = self.model._step(self.obs, S=self.states, M=self.dones)
            #actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_obs_acer.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_values.append(values)
            mb_values_a2c.append(values_a2c)
            mb_neglogpacs.append(neglogpacs)
            mb_neglogpacs_a2c.append(neglogpacs_a2c)
            mb_neglogpacs_acer.append(neglogpacs_acer)
            mb_dones.append(self.dones)
            mb_dones_acer.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            #print("PPO2 actions size: ")
            #print(np.shape(actions))
            obs, rewards, self.dones, infos = self.env.step(actions)
            self.obs[:] = obs
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            mb_rewards_a2c.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
        mb_obs_acer.append(np.copy(self.obs))
        mb_dones_acer.append(self.dones)
        #batch of steps to batch of rollouts
        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_acer = np.asarray(mb_obs_acer, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_a2c = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards_a2c = np.asarray(mb_rewards_a2c, dtype=np.float32).swapaxes(1, 0)
        mb_rewards_acer = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions_acer = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_actions_a2c = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions)
        #mb_values_a2c = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_dones_acer = np.asarray(mb_dones_acer, dtype=np.bool).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_masks_a2c = mb_dones_acer[:, :-1]
        mb_masks = mb_dones_acer # Used for statefull models like LSTM's to mask state when done
        mb_dones_acer = mb_dones_acer[:, 1:]

        #print("ppo2 self.obs shape at last_values: ")
        #print(np.shape(self.obs))


        agent = count.index(max(count))
        if agent != 1 and max(count) > 6144:
            params = tf.trainable_variables("ppo2_model")
            if agent == 0 and a2c_param:
                for i in range(len(params)):
                    #params[i].assign(a2c_param[i])
                    update = tf.assign(params[i],a2c_param[i])
                    self.model.sess.run(update)
                mb_values = np.asarray(mb_values_a2c, dtype=np.float32)
                mb_neglogpacs = np.asarray(mb_neglogpacs_a2c, dtype=np.float32)
            if agent == 2 and acer_param:
                for i in range(len(params)-2):
                    #params[i].assign(acer_param[i])
                    update = tf.assign(params[i],acer_param[i])
                    self.model.sess.run(update)
                mb_neglogpacs = np.asarray(mb_neglogpacs_acer, dtype=np.float32)


        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        #print("ppo2 last_values shape: ")
        #print(np.shape(last_values))
        #print(last_values)


        #print("ppo2 mb_actions_a2c size: ")
        #print(np.shape(mb_actions_a2c))
        #print("ppo2 mb_actions_acer size: ")
        #print(np.shape(mb_actions_acer))
        #print("ppo2 mb_actions_ppo2 size: ")
        #print(np.shape(mb_actions))
        """
        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_value = last_values.tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards_a2c, mb_dones_acer, last_value)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards_a2c[n] = rewards
        """

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            #print(mb_rewards[t])
            #print(nextvalues)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        if self.EVAL:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_states, epinfos)
        else:
            #print("ppo2 a2c mb_obs_a2c: " + str(np.shape(mb_obs_a2c)))
            #print("ppo2 a2c mb_obs_a2c[0]: " + str(np.shape(mb_obs_a2c[0])))
            #print("ppo2 a2c mb_states: " + str(mb_states))
            #print("ppo2 a2c mb_rewards_a2c: " + str(np.shape(mb_rewards_a2c)))
            #print("ppo2 a2c mb_masks_a2c: " + str(np.shape(mb_masks_a2c)))
            #print("ppo2 a2c mb_actions_a2c: " + str(np.shape(mb_actions_a2c)))
            #print("ppo2 a2c mb_values_a2c: " + str(np.shape(mb_values_a2c)))
            #print("ppo2 a2c epinfos: " + str(np.shape(epinfos)))
            """
            inds = np.arange(8192)
            for start in range(0, 8192, 2048):
                end = start + 2048
                mbinds = inds[start:end]
                slices = [arr[mbinds] for arr in (mb_obs_a2c, mb_rewards_a2c.flatten(), mb_masks_a2c.flatten(), mb_actions_a2c.flatten(), mb_values_a2c.flatten())]
                exp_a2c = [slices[0], mb_states, slices[1], slices[2], slices[3], slices[4]]
                if start == 0:
                    exp_a2c.append(epinfos)
                else:
                    exp_a2c.append(None)

                #self.q_exp[0].put(exp_a2c)
            """

            #exp_acer = [enc_obs, mb_obs_acer, mb_actions_acer, mb_rewards_acer, mb_mus, mb_dones_acer, mb_masks]
            #self.q_exp[2].put(exp_acer)

            ret = []
            ret.append([*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_states, epinfos])
            #print("ppo2 mb_actions shape: ")
            #print(np.shape(ret[0][3]))

            #while not self.q_exp[1].empty():
            #    exp_ppo2 = self.q_exp[1].get()
            #    ret.append(exp_ppo2)

        #return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
        #    mb_states, epinfos)
            return ret
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


