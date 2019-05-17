''' Simulator for ACM Multimedia 2019 Live Video Streaming Challenge
    Author Dan Yang
    Time 2019-01-31
'''
# import the env 
import fixed_env as fixed_env
import load_trace as load_trace
#import matplotlib.pyplot as plt
import time as tm
import ABR
import os
from eutils import *
# os.path.join("")
from replay_buffer import ReplayBuffer
from config import *
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs  码率可选择的范围，每一个码率对应的视频的信息数据对视不一样的
TARGET_BUFFER = [0.5, 1.0]  # seconds，目标可选择的buffer的大小

# MULTI_STEP = 3
# MODEL_PATH = "./model"
# UPDATE_TARGET_BY_EPISODE_END = 50
# UPDATE_TARGET_BY_EPISODE_START = 5
# UPDATE_TARGET_DECAY = 200
# UPDATE_TARGET_BY_EPISODE_RATE = (UPDATE_TARGET_BY_EPISODE_END - UPDATE_TARGET_BY_EPISODE_START) / UPDATE_TARGET_DECAY + 0.00001
# TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
class Experiment(object):
    def __init__(self,testing=False):
        mkdir_if_not_exist(MODEL_PATH)
        DEBUG = False
        random_seed = 2
        # Control the subdirectory where log files will be stored.
        LOG_FILE_PATH = './log/'
        # create result directory
        if not os.path.exists(LOG_FILE_PATH):
            os.makedirs(LOG_FILE_PATH)
        NETWORK_TRACE = 'fixed'
        VIDEO_TRACE = 'AsianCup_China_Uzbekistan'
        network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
        video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
        self.net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw,
                                        random_seed=random_seed,
                                        logfile_path=LOG_FILE_PATH,
                                        VIDEO_SIZE_FILE=video_trace_prefix,
                                        Debug=DEBUG)  # 视频相关的环境初始化，把load的所有的网络的数据输进去
        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_MAX,past_frame_len=FRAME_SKIP,multi_step=N_STEP)
        self.abr = ABR.Algorithm(self.net_env,self.replay_buffer)  # 加载ABR算法
        self.step_count = 0
        self.episode_count = 0
        self.target_net_update_count = 0
        self.testing = testing
        self.update_target_episode = UPDATE_TARGET_BY_EPISODE_START
        self.update_target_interval = UPDATE_TARGET_BY_EPISODE_START + UPDATE_TARGET_RATE
        # if testing:
        #     self.start_train()
        # else:
        #     self.start_train()

        # self.abr.Initial()

    def _run_epoch(self,i):
        # 每一次epoch里面设置固定的step步数
        # 每一个epoch里面可能包含多个episode,一个episode里面可能有多个step，计算step的总和和episode综合，如果超过了设定值计算
        step_left = EPOCH_LENGTH
        random_episode = True
        episode_in_epoch = 0
        step_in_epoch = 0
        reward_in_epoch = 0
        while step_left > 0:
            if self.step_count > BEGIN_RANDOM_STEP or self.testing:
                random_episode = False
            t0 = tm.time()
            episode_steps, episode_reward, average_loss, average_max_q = self.abr.run_episode(i,random_episode,self.testing)
            self.step_count += episode_steps
            if not random_episode:
                self.episode_count += 1
                episode_in_epoch += 1
                step_in_epoch += episode_steps
                reward_in_epoch += episode_reward
                step_left -= episode_steps
            t1 = tm.time()
            print("episode %d, episode step=%d,total_step=%d,time=%.2f,episode_reward=%.2f,average_loss=%.4f,average_q=%f"
                  % (self.episode_count, episode_steps,self.step_count,(t1 - t0),episode_reward,average_loss,average_max_q))
            self._update_target_net(random_episode)
        self._save_net()
        print("\n %s EPOCH FINISH %d,episode=%d,step=%d,average_step=%d,average_reward=%.2f \n\n" %
              (tm.strftime(TIME_FORMAT),
               i,
               self.episode_count,
               self.step_count,
               step_in_epoch // episode_in_epoch,
               reward_in_epoch / episode_in_epoch))
        return

    def _update_target_net(self,random_action=False):
        print(random_action)
        if not self.testing and self.episode_count == self.update_target_episode and not random_action:
            self.target_net_update_count += 1
            print("%s UPDATE TARGET NET,interval %.3f,update count %d\n" % (tm.strftime(TIME_FORMAT),self.update_target_interval,self.update_target_interval))
            self.update_target_episode = int(self.update_target_episode + self.update_target_interval)
            self.update_target_interval = min((self.update_target_interval + UPDATE_TARGET_RATE),UPDATE_TARGET_BY_EPISODE_END)
            self.abr.update_target_net()
        return

    def _save_net(self):
        if not self.testing:
            self.abr.save_params_to_file(model_path=MODEL_PATH,mark=tm.time())
            # self.abr.sa


    def run(self):
        for i in range(1, EPOCH_NUM):
            self._run_epoch(i)




# def train():
#
#     random_seed = 2
#     count = 0
#     trace_count = 1  # 正在计算的第几个视频
#     FPS = 25
#     frame_time_len = 0.04  # 帧速率
#     reward_all_sum = 0  # 总奖励
#     run_time = 0  # 运行时间
#
#
#     cnt = 0
#     # defalut setting
#     last_bit_rate = 0 # 上一帧的码率
#     bit_rate = 0
#     target_buffer = 0 # 目标buffer
#     latency_limit = 4  # 延迟限制
#
#     # QOE setting
#     """
#     比特率/帧速率−1.85∗拒绝−W1∗延迟−0.02∗交换机∗abs(比特率−最后一个比特率)−0.5∗跳过时间
#     """
#     reward_frame = 0
#     reward_all = 0
#     SMOOTH_PENALTY= 0.02  # 平滑惩罚系数
#     REBUF_PENALTY = 1.85  # 重新加载缓冲区的惩罚系数 拒绝
#     LANTENCY_PENALTY = 0.005
#     SKIP_PENALTY = 0.5  # 跳帧惩罚系数
#     # past_info setting
#     past_frame_num  = 7500  # 已经过去的帧信息
#     call_time_sum = 0
#     random_step = 10000
#     while True:
#         # random_cnt = 0
#         time, time_interval, send_data_size, chunk_len, \
#         rebuf, buffer_size, play_time_len, end_delay, \
#         cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
#         buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer, latency_limit)
#         replay_buffer.insert_sample(time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,
#                                     end_delay, cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len,
#                                     decision_flag, buffer_flag, cdn_flag, skip_flag, end_of_video)
#         # if random_cnt < random_step:
#
#         # else:
#
#             # break
#         # random_cnt += 1
#
#         break
#
#
#
#
#
#
#     return

def test(user_id):

    # -- Configuration variables --
    # Edit these variables to configure the simulator

    # Change which set of network trace to use: 'fixed' 'low' 'medium' 'high'
    NETWORK_TRACE = 'fixed'

    # Change which set of video trace to use.
    VIDEO_TRACE = 'AsianCup_China_Uzbekistan'

    # Turn on and off logging.  Set to 'True' to create log files.
    # Set to 'False' would speed up the simulator.
    DEBUG = False

    # Control the subdirectory where log files will be stored.
    LOG_FILE_PATH = './log/'
    
    # create result directory
    if not os.path.exists(LOG_FILE_PATH):
        os.makedirs(LOG_FILE_PATH)

    # -- End Configuration --
    # You shouldn't need to change the rest of the code here.

    network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
    video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'

    # load the trace
    """每个文件中的cooked_time和cooked_bw形成一个列表，append到总列表中，网络时延的情况"""
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
    # print(len(all_cooked_time))  # 20
    # print(len(all_cooked_time[0]))  # 5880
    # print(len(all_cooked_bw))
    # print(len(all_cooked_bw[0]))
    # print(all_file_names)
    # random_seed
    random_seed = 2
    count = 0
    trace_count = 1  # 正在计算的第几个视频
    FPS = 25
    frame_time_len = 0.04  # 帧速率
    reward_all_sum = 0  # 总奖励
    run_time = 0  # 运行时间
    #init 
    #setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=random_seed,
                                  logfile_path=LOG_FILE_PATH,
                                  VIDEO_SIZE_FILE=video_trace_prefix,
                                  Debug = DEBUG)  # 视频相关的环境初始化，把load的所有的网络的数据输进去
    ctx = try_gpu(0)
    input_sample = nd.ones((1,11),ctx,dtype=np.float32)
    abr = ABR.Algorithm() # 加载ABR算法
    abr_init = abr.Initial()

    BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs  码率可选择的范围，每一个码率对应的视频的信息数据对视不一样的
    TARGET_BUFFER = [0.5,1.0]   # seconds，目标可选择的buffer的大小
    # ABR setting
    RESEVOIR = 0.5  # ABR算法的配置，现在没有用
    CUSHION  = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0 # 上一帧的码率
    bit_rate = 0
    target_buffer = 0 # 目标buffer
    latency_limit = 4  # 延迟限制

    # QOE setting
    """
    比特率/帧速率−1.85∗拒绝−W1∗延迟−0.02∗交换机∗abs(比特率−最后一个比特率)−0.5∗跳过时间
    """
    reward_frame = 0
    reward_all = 0
    SMOOTH_PENALTY= 0.02  # 平滑惩罚系数
    REBUF_PENALTY = 1.85  # 重新加载缓冲区的惩罚系数 拒绝
    LANTENCY_PENALTY = 0.005
    SKIP_PENALTY = 0.5  # 跳帧惩罚系数
    # past_info setting
    past_frame_num  = 7500  # 已经过去的帧信息
    S_time_interval = [0] * past_frame_num  # 时间间隔
    S_send_data_size = [0] * past_frame_num  # 发送数据大小
    S_chunk_len = [0] * past_frame_num  #  块长度
    S_rebuf = [0] * past_frame_num  # 重新缓冲，拒绝，保证较少的拒绝事件
    S_buffer_size = [0] * past_frame_num  # 缓冲大小
    S_end_delay = [0] * past_frame_num  # 结束延迟
    S_chunk_size = [0] * past_frame_num  #
    S_play_time_len = [0] * past_frame_num  # 播放时间长度
    S_decision_flag = [0] * past_frame_num  # 决定的标志
    S_buffer_flag = [0] * past_frame_num  # buffer的标志
    S_cdn_flag = [0] * past_frame_num  # cdn的标志
    S_skip_time = [0] * past_frame_num  # 跳帧的时间
    # params setting
    call_time_sum = 0 
    while True:

        reward_frame = 0
        # input the train steps
        #if cnt > 5000:
            #plt.ioff()
        #    break
        #actions bit_rate  target_buffer
        # every steps to call the environment
        # time           : physical time 
        # time_interval  : time duration in this step
        # send_data_size : download frame data size in this step
        # chunk_len      : frame time len
        # rebuf          : rebuf time in this step          
        # buffer_size    : current client buffer_size in this step          
        # rtt            : current buffer  in this step          
        # play_time_len  : played time len  in this step          
        # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
        # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
        # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
        # cdn_flag       : If the True cdn has no frame to get 
        # end_of_video   : If the True ,which means the video is over.
        time,time_interval, send_data_size, chunk_len,\
               rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
                buffer_flag, cdn_flag, skip_flag,end_of_video = net_env.get_video_frame(bit_rate,target_buffer, latency_limit)
        # 给环境输入码率、目标的缓冲区大小和延迟到的限制得到关于视频播放的信息
        # S_info is sequential order
        # 把之前列表中存储的关于视频播放的信息的数据更新掉
        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_chunk_len.pop(0)
        S_buffer_size.pop(0)
        S_rebuf.pop(0)
        S_end_delay.pop(0)
        S_play_time_len.pop(0)
        S_decision_flag.pop(0)
        S_buffer_flag.pop(0)
        S_cdn_flag.pop(0)
        S_skip_time.pop(0)

        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_rebuf.append(rebuf)  # 拒绝
        S_end_delay.append(end_delay)
        S_play_time_len.append(play_time_len)
        S_decision_flag.append(decision_flag)
        S_buffer_flag.append(buffer_flag)
        S_cdn_flag.append(cdn_flag) 
        S_skip_time.append(skip_frame_time_len)

        # QOE setting
        # 结束延迟小于1的化，延迟的惩罚就为0.005，否则的化惩罚为0.01
        if end_delay <=1.0:
            LANTENCY_PENALTY = 0.005
        else:
            LANTENCY_PENALTY = 0.01
        # CDN flag是什么意思？？？？为什么会影响reward的计算
        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY  * end_delay - SKIP_PENALTY * skip_frame_time_len 
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)
        if decision_flag or end_of_video:
            # 如果是决策帧，或者是视频结束的时候要将reward加上一个比特率与上一个比特率之间的差值
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            # last_bit_rate
            last_bit_rate = bit_rate

            # ----------------- Your Algorithm ---------------------
            # which part is the algorithm part ,the buffer based ,
            # 如果缓冲足够，选择高质量的块，如果缓冲不够，选择低质量的块，如果没有rebuf,选择低目标缓冲？？？？
            # if the buffer is enough ,choose the high quality
            # if the buffer is danger, choose the low  quality
            # if there is no rebuf ,choose the low target_buffer
            cnt += 1
            timestamp_start = tm.time()
            # ABR算法运行根据当前时间和历史播放视频的情况输出码率、目标缓冲大小、延迟限制
            bit_rate, target_buffer, latency_limit = abr.run(time,
                    S_time_interval,
                    S_send_data_size,
                    S_chunk_len,
                    S_rebuf,
                    S_buffer_size, 
                    S_play_time_len,
                    S_end_delay,
                    S_decision_flag,
                    S_buffer_flag,
                    S_cdn_flag,
                    S_skip_time,
                    end_of_video, 
                    cdn_newest_id, 
                    download_id,
                    cdn_has_frame,
                    abr_init)
            # print(time)  # float
            # print(len(S_send_data_size))  # 7500
            # print(end_of_video) # boolean
            # print(cdn_newest_id) # int37836
            # print(download_id)  # int37799
            # print(len(cdn_has_frame))  # 5
            # print(len(cdn_has_frame[0]))  # 36,16,24,cdn_newest_id - download_id
            # print(abr_init)  # None
            timestamp_end = tm.time()
            # 算法的计算时间
            call_time_sum += timestamp_end - timestamp_start
            # -------------------- End --------------------------------
            
        if end_of_video:
            print("network traceID, network_reward, avg_running_time", trace_count, reward_all, call_time_sum/cnt)
            reward_all_sum += reward_all
            run_time += call_time_sum / cnt
            if trace_count >= len(all_file_names):  # 所有的视频都运行结束之后跳出循环
                break
            trace_count += 1
            cnt = 0
            
            call_time_sum = 0  # 整个视频的算法计算时间
            last_bit_rate = 0
            reward_all = 0
            bit_rate = 0
            target_buffer = 0

            S_time_interval = [0] * past_frame_num
            S_send_data_size = [0] * past_frame_num
            S_chunk_len = [0] * past_frame_num
            S_rebuf = [0] * past_frame_num
            S_buffer_size = [0] * past_frame_num
            S_end_delay = [0] * past_frame_num
            S_chunk_size = [0] * past_frame_num
            S_play_time_len = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_buffer_flag = [0] * past_frame_num
            S_cdn_flag = [0] * past_frame_num
            
        reward_all += reward_frame

    return [reward_all_sum / trace_count, run_time / trace_count]  # 平均奖励，平均运行时间
 
# a = test("aaa")
# print(a)

if __name__ == '__main__':
    runner = Experiment(False)
    runner.run()
