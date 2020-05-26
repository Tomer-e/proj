import tensorflow as tf
import numpy as np


class deepRMNetwork:

    def __init__(self,
                 input_height,
                 input_width,
                 output_length,
                 lr=0.1,
                 reward_decay=1.0,
                 decay=0.9,
                 epsilon=1e-9):
        self.input_height = input_height
        self.input_width = input_width
        self.output_length = output_length
        self.lr = lr
        self.reward_decay = reward_decay
        self.decay = decay
        self.epsilon = epsilon

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observations') #self.input_height*self.input_width
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num')
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name='actions_value')

            # self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            # self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            # self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")


        self.l_hid1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=20,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            bias_initializer=tf.constant_initializer(0),
            name='l_hid1'
        )

        self.all_act = tf.layers.dense(
            inputs=self.l_hid1,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0),
        )

        self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob')  # use softmax to convert to probability
        self.act_prob = tf.nn.softmax(self.all_act, name='act_prob')




        # self.optimizer = tf.train.GradientDescentOptimizer(self.lr)

    def choose_action(self, state):
        act_prob = self.sess.run(self.act_prob, feed_dict={self.states: state[None, :, :]})
        action = np.random.choice(range(act_prob.shape[-1]), p=act_prob)
        return action

    def learn(self, states, actions, rewards):
        self.sess.run(self.train_op, feed_dict={
            self.states: states,
            self.actions: actions,
            self.rewards: self._discount_and_norm_rewards(rewards, self.reward_decay)
        })

    def _discount_and_norm_rewards(self, rewards, gamma):
        # discount episode rewards
        discounted_experience_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_experience_rewards[t] = running_add

        # normalize episode rewards
        discounted_experience_rewards -= np.mean(discounted_experience_rewards)
        discounted_experience_rewards /= np.std(discounted_experience_rewards)
        return discounted_experience_rewards
