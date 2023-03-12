import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces
from baselines.a2c.utils import discount_with_dones
import tensorflow as tf
import sys
import math

#tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_v2_behavior()

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, q_exp, q_model, model_a2c, model_ppo2):
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
        self.models = [model_a2c, model_ppo2, model]

        #self.model_a2c = model_a2c
        #self.model_ppo2 = model_ppo2

    def run(self):

        ppo2_param = None
        while not self.q_model[1].empty():
            ppo2_param = self.q_model[1].get()
        if ppo2_param:
            #print("ppo2_param received: ")
            #print(ppo2_param)
            params = tf.trainable_variables("ppo2_model")
            #print("params current ppo2 model: ")
            #print(params)
            for i in range(len(params)):
                #params[i].assign(ppo2_param[i])
                update = tf.assign(params[i],ppo2_param[i])
                self.models[1].sess.run(update)
                #print("params " + str(i) + " :")
                #print(self.models[1].sess.run(params[i]))
            #print("params ppo2 model after assign: ")
            #print(params)
        a2c_param = None
        while not self.q_model[0].empty():
            a2c_param = self.q_model[0].get()
        if a2c_param:
            #print("a2c_param received: ")
            #print(a2c_param)
            params = tf.trainable_variables("a2c_model")
            #print("params current a2c model: ")
            #print(params)
            for i in range(len(params)):
                #params[i].assign(a2c_param[i])
                update = tf.assign(params[i],a2c_param[i])
                self.models[0].sess.run(update)
                #print("params " + str(i) + " :")
                #print(self.models[0].sess.run(params[i]))
                #tf.Print(params[i], [params[i]])
            #print("params a2c model after assign: ")
            #print(params)


        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs_acer, mb_actions, mb_mus, mb_mus_a2c, mb_mus_ppo2, mb_dones, mb_rewards = [], [], [], [], [], [], []
        #mb_states = self.states
        epinfos = []
        count = [0,0,0]
        value_sum = [0,0,0]
        for _ in range(self.nsteps):#+1):
            #print("ACER self.obs: ")
            #print(np.shape(self.obs))

            action_list = []
            value_list = []
            state_list = []
            likelihood_list = []
            mus_list = []
            for k in range(3):
                _, tmp1, _, tmp3 = self.models[k].step(self.obs, S=self.states, M=self.dones)
                tmp0, tmp4, tmp2, tmp5 = self.models[k]._step(self.obs, S=self.states, M=self.dones)
                action_list.append(tmp0)
                #print("agent " + str(k) + " selected action and likelihood:")
                #print(tmp0)
                value_list.append(tmp5)
                state_list.append(tmp2)
                likelihood_list.append(tmp3)
                #print(tmp3)
                mus_list.append(tmp4)
                #print(tmp4)
            value_sum[0] += sum(value_list[0])
            value_sum[1] += sum(value_list[1])
            value_sum[2] += sum([sum([tmp1[i,j]*mus_list[2][i,j] for j in range(6)]) for i in range(4)])
            #print("acer action_list: ")
            #print(action_list)
            #print("acer likelihood_list: ")
            #print(likelihood_list)
            for k in range(4):
                flag = 0
                temp = [likelihood_list[0][k],likelihood_list[1][k],likelihood_list[2][k]]
                temp_min = min(temp)
                index = 2
                threshold0 = -1 * math.log(1/self.nact) * 0.85
                threshold1 = -1 * math.log(1/self.nact) * 0.75
                #print("threshold = " + str(threshold))
                for j in range(3):
                    if likelihood_list[j][k] < threshold0 and (value_sum[j] == max(value_sum)):
                        index = j
                        flag = 1
                if temp_min < threshold1 and flag == 0:
                    index = temp.index(temp_min)
                #print("selected agent:")
                #print(index)
                count[index] += 1
                action_list[0][k] = action_list[index][k]
                #value_list[0][k] = value_list[index][k]
                #state_list[0][k] = state_list[index][k]
                #likelihood_list[0][k] = likelihood_list[index][k]
                #mus_list[0][k] = mus_list[index][k]
                #likelihood_list[0][k] = -1 * math.log(mus_list[2][k][action_list[0][k]])
            #print("acer action_list after get max: ")
            #print(action_list)
            #print("acer likelihood_list after get max: ")
            #print(likelihood_list)

            actions = action_list[0]
            #values = value_list[2]
            states = state_list[0]
            #neglogpacs = likelihood_list[0]
            mus_a2c = mus_list[0]
            mus_ppo2 = mus_list[1]
            mus = mus_list[2]


            #actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones)
            #_, values, _, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            #print("acer step values: " + str(values))
            #mb_obs.append(np.copy(self.obs))
            mb_obs_acer.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_mus_a2c.append(mus_a2c)
            mb_mus_ppo2.append(mus_ppo2)
            mb_dones.append(self.dones)
            #mb_dones_ppo2.append(self.dones)
            """
            tmp = np.zeros(4)
            for i in range(4):
                for j in range(6):
                    tmp[i] += values[i,j] * mus[i,j]
            values = tmp
            mb_values.append(values)
            """
            #mb_neglogpacs.append(neglogpacs)
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

        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        agent = count.index(max(count))
        if agent != 2 and max(count) > 40:
            print("acer count: ")
            print(count)
            params = tf.trainable_variables("acer_model")
            #print("acer_model: ")
            #print(params)
            if agent == 1 and ppo2_param:
                #print("ppo2_param received: ")
                #print(ppo2_param)
                print("acer use ppo2 model")
                for i in range(len(params)-4):
                    #params[i].assign(ppo2_param[i])
                    update = tf.assign(params[i],ppo2_param[i])
                    self.model.sess.run(update)
                    #print("params " + str(i) + " :")
                    #tf.Print(params[i], [params[i]])
                #print("acer_model after assign: ")
                #print(params)
                mb_mus = np.asarray(mb_mus_ppo2, dtype=np.float32).swapaxes(1, 0)
            if agent == 0 and a2c_param:
                #print("a2c_model received: ")
                #print(a2c_param)
                print("acer use a2c model")
                for i in range(len(params)-4):
                    #params[i].assign(a2c_param[i])
                    update = tf.assign(params[i],a2c_param[i])
                    self.model.sess.run(update)
                    #print("params " + str(i) + " :")
                    #tf.print(params[i], [params[i]])
                #print("acer_model after assign: ")
                #print(params)
                mb_mus = np.asarray(mb_mus_a2c, dtype=np.float32).swapaxes(1, 0)


        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        #enc_obs = np.asarray(enc_obs[0:511], dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_acer = np.asarray(mb_obs_acer, dtype=self.obs_dtype).swapaxes(1, 0)
        #mb_obs_acer = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        #mb_obs_ppo2 = np.asarray(mb_obs, dtype=self.obs.dtype)
        #mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape_acer)
        mb_actions_acer = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        #mb_actions_acer = np.asarray(mb_actions[0:511], dtype=self.ac_dtype).swapaxes(1, 0)
        #mb_actions_ppo2 = np.asarray(mb_actions)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_rewards_acer = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        #mb_rewards_acer = np.asarray(mb_rewards[0:511], dtype=np.float32).swapaxes(1, 0)
        #mb_rewards_ppo2 = np.asarray(mb_rewards, dtype=np.float32)
        #mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        #mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        #mb_mus = np.asarray(mb_mus[0:511], dtype=np.float32).swapaxes(1, 0)
        #mb_values_ppo2 = np.asarray(mb_values, dtype=np.float32)
        #mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        #mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)

        #mb_dones_ppo2 = np.asarray(mb_dones_ppo2, dtype=np.bool)
        mb_dones_acer = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        #mb_dones_acer = np.asarray(mb_dones_ppo2, dtype=np.bool).swapaxes(1, 0)
        mb_masks_acer = mb_dones_acer # Used for statefull models like LSTM's to mask state when done
        mb_dones_acer = mb_dones_acer[:, 1:]
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
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
        """
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
        """

        #exp_a2c = [mb_obs, None, mb_rewards.flatten(), mb_masks.flatten(), mb_actions.reshape(self.batch_action_shape), mb_values.flatten(), epinfos]

        #ll = list(map(sf01, (mb_obs_ppo2, mb_returns, mb_dones_ppo2, mb_actions_ppo2, mb_values_ppo2, mb_neglogpacs)))
        #exp_ppo2 = [ll[0], ll[1], ll[2], ll[3], ll[4], ll[5], None, epinfos]

        #self.q_exp[0].put(exp_a2c)
        #self.q_exp[1].put(exp_ppo2)

        ret = []
        #print("mb_obs_acer shape before return: ")
        #print(np.shape(mb_obs_acer))
        ret.append([enc_obs, mb_obs_acer, mb_actions_acer, mb_rewards_acer, mb_mus, mb_dones_acer, mb_masks_acer])

        #while not self.q_exp[2].empty():
        #    exp_acer = self.q_exp[2].get()
        #    ret.append(exp_acer)

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
