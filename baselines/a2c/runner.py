import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
import tensorflow as tf

#tf.compat.v1.enable_eager_execution()

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, q_exp, q_model, model_ppo2, model_acer, nsteps=5, gamma=0.99):
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
        self.models = [model, model_ppo2, model_acer]
        #self.models.append(model)
        #self.models.append(model_ppo2)
        #self.models.append(model_acer)
        #self.model_ppo2 = model_ppo2
        #self.model_acer = model_acer
        self.lam=0.95

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
            #print("params ppo2 model after assign: ")
            #print(params)
        acer_param = None
        while not self.q_model[2].empty():
            acer_param = self.q_model[2].get()
        if acer_param:
            #print("acer_param received: ")
            #print(acer_param)
            params = tf.trainable_variables("acer_model")
            #print("params current acer model: ")
            #print(params)
            for i in range(len(params)-2):
                #params[i].assign(acer_param[i])
                update = tf.assign(params[i],acer_param[i])
                self.models[2].sess.run(update)
            #print("params acer model after assign: ")
            #print(params)

        # We initialize the lists that will contain the mb of experiences
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_obs_acer, mb_rewards, mb_actions, mb_values, mb_dones, mb_dones_ppo2, mb_mus, mb_neglogpacs = [], [], [], [], [],[],[],[],[]
        mb_states = self.states
        epinfos = []
        #print("A2C self.obs: ")
        #print(np.shape(self.obs))
        count = [0,0,0]
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on initi

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
                index = temp.index(min(temp))
                count[index] += 1
                action_list[0][k] = action_list[index][k]
                value_list[0][k] = value_list[index][k]
                #state_list[0][k] = state_list[index][k]
                likelihood_list[0][k] = likelihood_list[index][k]
                mus_list[0][k] = mus_list[index][k]


            #actions, values, states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            #_, mus, _ = self.model._step(self.obs, S=self.states, M=self.dones)


            actions = action_list[0]
            values = value_list[0]
            states = state_list[0]
            neglogpacs = likelihood_list[0]
            mus = mus_list[0]


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


        agent = count.index(max(count))
        if agent != 0 and max(count) > 15:
            params = tf.trainable_variables("a2c_model")
            #print("a2c_model: ")
            #print(params)
            if agent == 1 and ppo2_param:
                #print("ppo2_param received: ")
                #print(ppo2_param)
                for i in range(len(params)):
                    #params[i].assign(ppo2_param[i])
                    update = tf.assign(params[i],ppo2_param[i])
                    self.model.sess.run(update)
                    #print("params " + str(i) + ":")
                    #print(params[i].numpy())
                #print("a2c_model after assign: ")
                #print(params)
            if agent == 2 and acer_param:
                #print("acer_model received: ")
                #print(acer_param)
                for i in range(len(params)-2):
                    #params[i].assign(acer_param[i])
                    update = tf.assign(params[i],acer_param[i])
                    self.model.sess.run(update)
                    #print("params " + str(i) + ":")
                    #print(params[i].numpy())
                #print("a2c_model after assign: ")
                #print(params)


        # Batch of steps to batch of rollouts
        enc_obs = np.asarray(enc_obs[0:511], dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_acer = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs_ppo2 = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards_acer = np.asarray(mb_rewards[0:511], dtype=np.float32).swapaxes(1, 0)
        #print("mb_rewards_acer shape: " + str(np.shape(mb_rewards_acer)))
        #print("mb_rewards shape: " + str(np.shape(mb_rewards)))
        mb_rewards_ppo2 = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions_acer = np.asarray(mb_actions[0:511], dtype=self.ac_dtype).swapaxes(1, 0)
        mb_actions_ppo2 = np.asarray(mb_actions)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values_ppo2 = np.asarray(mb_values, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_mus = np.asarray(mb_mus[0:511], dtype=np.float32).swapaxes(1, 0)

        mb_dones_ppo2 = np.asarray(mb_dones_ppo2, dtype=np.bool)
        mb_dones_acer = np.asarray(mb_dones_ppo2, dtype=np.bool).swapaxes(1, 0)
        mb_masks_acer = mb_dones_acer
        mb_dones_acer = mb_dones_acer[:, 1:]
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
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
                #print("rewards_tolist shape: " + str(len(rewards)))
                dones = dones.tolist()
                #print("dones_tolist shape: " + str(len(dones)))
                if dones[-1] == 0:
                    #print("here")
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    #print("there")
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                #print("rewards shape: " + str(np.shape(rewards)))
                #print("mb_rewards shape: " + str(np.shape(mb_rewards)))
                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()


        exp_acer = [enc_obs, mb_obs_acer, mb_actions_acer, mb_rewards_acer, mb_mus, mb_dones_acer, mb_masks_acer]

        ll = list(map(sf01, (mb_obs_ppo2, mb_returns, mb_dones_ppo2, mb_actions_ppo2, mb_values_ppo2, mb_neglogpacs)))
        exp_ppo2 = [ll[0], ll[1], ll[2], ll[3], ll[4], ll[5], mb_states, epinfos]

        #self.q_exp[1].put(exp_ppo2)
        #self.q_exp[2].put(exp_acer)

        ret = []
        ret.append([mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos])

        #while not self.q_exp[0].empty():
        #    exp_a2c = self.q_exp[0].get()
        #    ret.append(exp_a2c)

        #return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos
        return ret

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
