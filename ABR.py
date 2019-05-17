# import tensorflow as tf
import numpy as np
from mxnet.gluon import *
from mxnet import nd,init
from config import *
import mxnet as mx
import dueling_dqn
import time as tm
from eutils import *

#NN_MODEL = "./submit/results/nn_model_ep_18200.ckpt" # model path settings
TARGET_BUFFER = [2.0 , 3.0]
GPU_INDEX = 0
DTYPE = np.float32

INPUT_SAMPLE = nd.ones((1,11),)

LEARNING_RATE = 0.0001
EPSILON_MIN = 0.1
EPSILON_START = 1.0
EPSILON_DECAY = 1000000
K_FRAME = 10
ACTION_NUM = 8
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TRAIN_PER_STEP = 10
MULTI_STEP = 3

BIT_RATE = [500.0,850.0,1200.0,1850.0]

class Algorithm:
     def __init__(self,net_env,replay_buffer):
     # fill your self params
         self.buffer_size = 0
         self.last_bitrate = 0
         self.Initial()
         self.net_env = net_env
         self.replay_buffer = replay_buffer


     # Intial
     def Initial(self):
     # Initail your session or somethingself.target_net
     # restore neural net parameters
         self.buffer_size = 0
         self.ctx = try_gpu(GPU_INDEX)
         self.dtype = DTYPE
         INPUT_SAMPLE = nd.random_uniform(0,1,(1, FRAME_SKIP, 11), self.ctx, self.dtype)
         self.target_net = self.get_net(INPUT_SAMPLE)
         self.policy_net = self.get_net(INPUT_SAMPLE)
         model_file = None
         if model_file is not None:
             print('%s: read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
             self.policy_net.load_params(model_file, ctx=self.ctx)
         self.update_target_net()
         # adagrad
         self.trainer = Trainer(self.policy_net.collect_params(),
                                optimizer=mx.optimizer.RMSProp(LEARNING_RATE, 0.95, 0.95))
         self.loss_func = loss.L2Loss()

         self.epsilon = EPSILON_START
         # self.epsilon_min = EPSILON_MIN
         # self.epsilon_rate = (EPSILON_START - EPSILON_MIN) * 1.0 / EPSILON_DECAY
         self.epsilon_min = EPSILON_MIN
         self.epsilon_rate = (EPSILON_START - EPSILON_MIN) / EPSILON_DECAY
         self.rng = np.random.RandomState(int(time() * 1000) % 100000000)


     def update_target_net(self):
         self.copy_params(self.policy_net, self.target_net)
         return

     def run_episode(self,epoch,random_state,testing):
         episode_step = 0
         episode_reward = 0.0
         train_count = 0
         loss_sum = 0
         q_sum = 0
         q_count = 0

         while True:
             if random_state:
                 action, max_q = self.choose_action(random_state, testing, None)
                 # print("random action:%d" % action)
             else:
                 st = self.replay_buffer.get_current_state()
                 st = nd.array(st,ctx=self.ctx,dtype=self.dtype).reshape((1,-1,11))
                 # st.reshape(())
                 action, max_q = self.choose_action(random_state, testing, st)
                 # print("网络输出action:%d" % action)
             bitrate, target_buffer, latency = self.action_to_submit(action)
             # print(bitrate,target_buffer,latency)
             time, time_interval, send_data_size, chunk_len, \
             rebuf, buffer_size, play_time_len, end_delay, \
             cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
             buffer_flag, cdn_flag, skip_flag, end_of_video = self.net_env.get_video_frame(bitrate, target_buffer,
                                                                                           latency)
             # print("收到客户端信息")

             if end_of_video <= 1.0:
                 LANTENCY_PENALTY = 0.005
             else:
                 LANTENCY_PENALTY = 0.01
             if not cdn_flag:
                 reward_frame = FRAME_SKIP * float(BIT_RATE[bitrate]) / 1000 - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
             else:
                 reward_frame = -(REBUF_PENALTY * rebuf)
             if decision_flag or end_of_video:
                 reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bitrate] - BIT_RATE[self.last_bitrate]) / 1000)
             # print("reward %.3f" % reward_frame)
             self.last_bitrate = bitrate

             self.replay_buffer.insert_sample(time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,
                                  end_delay, cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len,
                                  decision_flag, buffer_flag, cdn_flag, skip_flag, end_of_video, reward_frame, action)
             # print("save buffer")
             episode_step += 1
             episode_reward += reward_frame
             if end_of_video:
                 # print("end of video")
                 break
             if not testing and episode_step % TRAIN_PER_STEP == 0 and not random_state:
                 state, s_, actions, rewards = self.replay_buffer.get_batch(16)
                 loss = self.train_policy_net(state,actions,rewards,s_)
                 loss_sum += loss
                 train_count += 1


         return episode_step, episode_reward, loss_sum / (train_count + 0.0001), q_sum / (q_count + 0.000001)

     def train_policy_net(self,states,actions,rewards,next_states):
         batch_size = actions.shape[0]
         s = states.shape
         states = nd.array(states,ctx=self.ctx,dtype=self.dtype)
         actions = nd.array(actions[:,0],ctx=self.ctx)
         rewards = nd.array(rewards[:,0],ctx=self.ctx)
         next_states = nd.array(next_states,ctx=self.ctx,dtype=self.dtype)

         next_qs = self.target_net(next_states)
         next_q_out = nd.max(next_qs,axis=1)

         target = rewards + next_q_out * 0.99 ** MULTI_STEP

         with autograd.record():
             current_qs = self.policy_net(states)
             current_q = nd.pick(current_qs,actions,1)
             loss = self.loss_func(target,current_q)
         loss.backward()
         self.trainer.step(16)
         total_loss = loss.mean().asscalar()
         return total_loss

     def save_params_to_file(self,model_path,mark):
         time_mark = tm.time()
         filename = model_path + '/net_' + str(mark) + '_' + str(time_mark) + '.model'
         self.policy_net.save_params(filename)
         print(tm.strftime(TIME_FORMAT), 'save model success:',filename)
         files = getNewestFile(model_path)
         if len(files) > 5:
             tmp = files[5:]
             for f in tmp:
                 if os.path.exists(model_path + "/" + f):
                     os.remove(model_path + "/" + f)
                     print(f + "is deleted.")



     def get_net(self, input_sample):
         if IS_DUELING:
             net = dueling_dqn.DuelingDQN()
             net.initialize(init.Xavier(), ctx=self.ctx)
         else:
             net = dueling_dqn.OriginDQN()
             net.initialize(init.Xavier(), ctx=self.ctx)
         net(input_sample)
         return net

     def choose_action(self, random_action, testing, st):

         self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
         max_q = None
         random_num = self.rng.rand(0,1)
         if random_action or ((not testing) and random_num < self.epsilon):
             action = self.rng.randint(0,ACTION_NUM)
         else:
             # print(st.shape)
             out = self.policy_net(st)
             max_index = nd.argmax(out, axis=1)
             action = int(max_index.astype(np.int).asscalar())
             max_q = out[0, action].asscalar()
             # print("网络输出action:%d" % action)
         # bitrate = action % 4
         # target_buffer = action // 4
         return action, max_q

     def action_to_submit(self,action):
         bitrate = action % 4
         target_buffer = action // 4
         latency = 4
         return bitrate,target_buffer,latency


     #Define your al
     # def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
     #         S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag,S_skip_time,
     #         end_of_video, cdn_newest_id,download_id,cdn_has_frame,IntialVars):
     #     # state = np.empty(shape=(len(S_time_interval),11),dtype=np.float32)
     #     buffer = [S_time_interval,S_send_data_size,S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
     #                  S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time]
     #     buffer = np.array(buffer,dtype=self.dtype)
     #     buffer = buffer.transpose((1,0))
     #     # bit_rate, target_buffer, max_q = self.choose_action(buffer,True,False)
     #     latency_limit = 4
     #     return bit_rate, target_buffer, latency_limit




     def get_params(self):
     # get your params
        your_params = []
        return your_params

     def copy_params(self, src_net, dst_net):
         ps_src = src_net.collect_params()
         ps_dst = dst_net.collect_params()
         prefix_length = len(src_net.prefix)
         for k, v in ps_src.items():
             k = k[prefix_length:]
             v_dst = ps_dst.get(k)
             v_dst.set_data(v.data())


def try_gpu(idx=0):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    if idx < 0:
        print('use CPU.')
        ctx = mx.cpu()
    else:
        try:
            ctx = mx.gpu(idx)
            _ = nd.array([0], ctx=ctx)
            print('Got GPU[%d] success.' % idx)
        except:
            print('Got GPU[%d] failed, use CPU.' % idx)
            ctx = mx.cpu()
    return ctx