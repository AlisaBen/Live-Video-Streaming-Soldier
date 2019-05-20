# import tensorflow as tf
import numpy as np
from mxnet.gluon import *
from mxnet import nd,init
# from config import *
import mxnet as mx
import dueling_dqn
import time as tm
from eutils import *
from replay_buffer import ReplayBuffer
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
#NN_MODEL = "./submit/results/nn_model_ep_18200.ckpt" # results path settings
TARGET_BUFFER = [2.0 , 3.0]
GPU_INDEX = 0
DTYPE = np.float32

IS_DOUBLE = False
IS_DUELING = False

# GPU_INDEX = 0

LEARNING_RATE = 0.0001
EPSILON_MIN = 0.1
EPSILON_START = 1.0
EPSILON_DECAY = 1000000
K_FRAME = 10
ACTION_NUM = 8
# TIME_FORMAT = %Y-%m-%d %H:%M:%S
TRAIN_PER_STEP = 10
MULTI_STEP = 3
BUFFER_MAX = 500
FRAME_SKIP = 10
MODEL_PATH = "./results"
UPDATE_TARGET_BY_EPISODE_END = 50
UPDATE_TARGET_BY_EPISODE_START = 5
UPDATE_TARGET_DECAY = 200

EPOCH_NUM = 10
EPOCH_LENGTH = 1000
BEGIN_RANDOM_STEP = 100
N_STEP = 3

SMOOTH_PENALTY= 0.02
LANTENCY_PENALTY = 0.005
REBUF_PENALTY = 1.85
SKIP_PENALTY = 0.5
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# TRAIN_PER_STEP = 10
# MULTI_STEP = 3

BIT_RATE = [500.0,850.0,1200.0,1850.0]
# MODEL_FILE = './submit/results/net_dqn_1558180924.3550863.model'
MODEL_FILE = './model/net_dqn_1558273425.4275875.model'
# MODEL_FILE = None
class Algorithm:
     def __init__(self):
         self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_MAX, past_frame_len=FRAME_SKIP, multi_step=N_STEP)

     # Intial
     def Initial(self):
     # Initail your session or somethingself.target_net
     # restore neural net parameters
         # self.buffer_size = 0
         self.ctx = try_gpu(GPU_INDEX)
         self.frame_cnt = 0
         self.train_count = 0
         self.loss_sum = 0

         self.q_count = 0
         self.q_sum = 0
         self.dtype = DTYPE
         INPUT_SAMPLE = nd.random_uniform(0,1,(1, FRAME_SKIP, 11), self.ctx, self.dtype)
         self.target_net = self.get_net(INPUT_SAMPLE)
         self.policy_net = self.get_net(INPUT_SAMPLE)

         if MODEL_FILE is not None:
             print('%s: read trained results from [%s]' % (tm.strftime("%Y-%m-%d %H:%M:%S"), MODEL_FILE))
             self.policy_net.load_params(MODEL_FILE, ctx=self.ctx)
         self.update_target_net()
         # adagrad
         self.trainer = Trainer(self.policy_net.collect_params(),
                                optimizer=mx.optimizer.RMSProp(LEARNING_RATE, 0.95, 0.95))
         self.loss_func = loss.L2Loss()

         self.epsilon = EPSILON_START
         self.epsilon_min = EPSILON_MIN
         self.epsilon_rate = (EPSILON_START - EPSILON_MIN) / EPSILON_DECAY
         self.rng = np.random.RandomState(int(time() * 1000) % 100000000)


     def update_target_net(self):
         self.copy_params(self.policy_net, self.target_net)
         return

     def calculate_reward(self,end_of_video,cdn_flag,rebuf,end_delay,skip_frame_time_len,decision_flag,bitrate,last_bitrate,frame_time_len):
         if end_of_video <= 1.0:
             LANTENCY_PENALTY = 0.005
         else:
             LANTENCY_PENALTY = 0.01
         if not cdn_flag:
             reward_frame = frame_time_len * float(BIT_RATE[
                                                   bitrate]) / 1000 - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
         else:
             reward_frame = -(REBUF_PENALTY * rebuf)
         if decision_flag or end_of_video:
             reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bitrate] - BIT_RATE[last_bitrate]) / 1000)
         return reward_frame

     def run_frame(self,time, time_interval, send_data_size, chunk_len, \
                rebuf, buffer_size, play_time_len, end_delay, \
                cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
                buffer_flag, cdn_flag, skip_flag, end_of_video,action,last_action,frame_time_len):

         bitrate, target_buffer, latency = self.action_to_submit(action)
         last_bitrate,_,_ = self.action_to_submit(last_action)
         reward_frame = self.calculate_reward(end_of_video,cdn_flag,rebuf,end_delay,skip_frame_time_len,decision_flag,bitrate,last_bitrate,frame_time_len)
         self.replay_buffer.insert_sample(time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,end_delay, cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len,decision_flag, buffer_flag, cdn_flag, skip_flag, end_of_video, reward_frame, action)
         st = self.replay_buffer.get_current_state()
         st = nd.array(st, ctx=self.ctx, dtype=self.dtype).reshape((1, FRAME_SKIP, -1))
         action, max_q = self.choose_action(False, False, st)
         bit_rate, target_buffer, latency_limit = self.action_to_submit(action)
         self.frame_cnt += 1
         if max_q is not None:
             self.q_count += 1
             self.q_sum += max_q
         if  self.frame_cnt % TRAIN_PER_STEP == 0:
             state, s_, actions, rewards = self.replay_buffer.get_batch(16)
             loss = self.train_policy_net(state, actions, rewards, s_)
             self.train_count += 1
             self.loss_sum += loss
         # fixme 视频结束的时候是否需要清零
         if end_of_video:
             average_loss = self.loss_sum / (self.train_count + 0.0001)
             average_q = self.q_sum / (self.q_count + 0.000001)
             self.loss_sum = 0
             self.train_count = 0
             self.q_count = 0
             self.q_sum = 0
         else:
             average_loss = 0
             average_q = 0
         return reward_frame,bit_rate, target_buffer, latency_limit,action,average_loss,average_q

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
         print(tm.strftime(TIME_FORMAT), 'save results success:',filename)
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
         random_num = self.rng.rand()
         if random_action or ((not testing) and random_num < self.epsilon):
             action = self.rng.randint(0,ACTION_NUM)
         else:
             out = self.policy_net(st)
             max_index = nd.argmax(out, axis=1)
             action = int(max_index.astype(np.int).asscalar())
             max_q = out[0, action].asscalar()
         return action, max_q

     def action_to_submit(self,action):
         bit_rate = action % 4
         target_buffer = action // 4
         latency_limit = 4
         return bit_rate, target_buffer, latency_limit


     #Define your al
     def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
             S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag,S_skip_time,
             end_of_video, cdn_newest_id,download_id,cdn_has_frame,IntialVars):
         # state = np.empty(shape=(len(S_time_interval),11),dtype=np.float32)
         S_end_of_video = [0] * FRAME_SKIP
         S_end_of_video[-1] = end_of_video
         state = [S_time_interval[-FRAME_SKIP:],S_send_data_size[-FRAME_SKIP:],S_chunk_len[-FRAME_SKIP:], S_buffer_size[-FRAME_SKIP:], S_rebuf[-FRAME_SKIP:],
                      S_end_delay[-FRAME_SKIP:],  S_play_time_len[-FRAME_SKIP:],S_decision_flag[-FRAME_SKIP:], S_cdn_flag[-FRAME_SKIP:],S_skip_time[-FRAME_SKIP:],S_end_of_video]

         state = nd.array(state,dtype=self.dtype).transpose((1,0)).reshape((1,FRAME_SKIP,-1))
         # print(state.shape)

         action, max_q = self.choose_action(False,True,state)
         # print(action)
         bit_rate, target_buffer, latency_limit = self.action_to_submit(action)
         print(bit_rate, target_buffer, latency_limit)

         return bit_rate, target_buffer, latency_limit

     def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
             S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
             download_id, cdn_has_frame, IntialVars):

         # If you choose the marchine learning
         '''state = []
         state[0] = ...
         state[1] = ...
         state[2] = ...
         state[3] = ...
         state[4] = ...
         decision = actor.predict(state).argmax()
         bit_rate, target_buffer = decison//4, decison % 4 .....
         return bit_rate, target_buffer'''

         # If you choose BBA
         RESEVOIR = 0.5
         CUSHION = 1.5

         if S_buffer_size[-1] < RESEVOIR:
             bit_rate = 0
         elif S_buffer_size[-1] >= RESEVOIR + CUSHION and S_buffer_size[-1] < CUSHION + CUSHION:
             bit_rate = 2
         elif S_buffer_size[-1] >= CUSHION + CUSHION:
             bit_rate = 3
         else:
             bit_rate = 1

         target_buffer = 0
         latency_limit = 4

         return bit_rate, target_buffer, latency_limit




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