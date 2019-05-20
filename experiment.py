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

class Experiment(object):
    def __init__(self,testing=False):
        mkdir_if_not_exist(MODEL_PATH)
        self.load_environment()
        self.abr = ABR.Algorithm()  # 加载ABR算法
        self.abr.Initial()
        # self.step_count = 0
        self.episode_count = 0
        self.target_net_update_count = 0
        self.testing = testing
        self.update_target_episode = UPDATE_TARGET_BY_EPISODE_START
        self.update_target_interval = UPDATE_TARGET_BY_EPISODE_START + UPDATE_TARGET_RATE

    def load_environment(self):
        DEBUG = False
        random_seed = 2
        # Control the subdirectory where log files will be stored.
        LOG_FILE_PATH = './log/'
        # create result directory
        if not os.path.exists(LOG_FILE_PATH):
            os.makedirs(LOG_FILE_PATH)
        # NETWORK_TRACE = 'fixed'
        VIDEO_TRACE = 'AsianCup_China_Uzbekistan'
        # network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
        # all_cooked_time, all_cooked_bw, self.all_file_names = load_trace.load_trace(network_trace_dir)
        network_trace = ['fixed', 'high']
        video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
        network_trace_dir_list = ['./dataset/network_trace/' + trace + '/' for trace in network_trace]
        all_cooked_time, all_cooked_bw, self.all_file_names = load_trace.load_trace_list(network_trace_dir_list)
        self.net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw,
                                        random_seed=random_seed,
                                        logfile_path=LOG_FILE_PATH,
                                        VIDEO_SIZE_FILE=video_trace_prefix,
                                        Debug=DEBUG)  # 视频相关的环境初始化，把load的所有的网络的数据输进去
        return

    def _run_epoch(self,i):
        # 每一次epoch里面设置固定的step步数
        # 每一个epoch里面可能包含多个episode,一个episode里面可能有多个step，计算step的总和和episode综合，如果超过了设定值计算
        trace_count = 1
        frame_time_len = 0.04
        reward_all_sum = 0
        # loss_all_sum = 0
        # q_all_sum = 0
        run_time = 0
        frame_all_sum = 0

        # end_of_video = False
        while True:
            # 一个epoch
            cnt = 0
            action = 0
            bit_rate, target_buffer, latency_limit = self.abr.action_to_submit(action)
            last_action = action

            reward_all = 0
            call_time_sum = 0
            while True:
                # 一个视频，一个episode
                time, time_interval, send_data_size, chunk_len, \
                rebuf, buffer_size, play_time_len, end_delay, \
                cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
                buffer_flag, cdn_flag, skip_flag, end_of_video = self.net_env.get_video_frame(bit_rate, target_buffer,latency_limit)
                timestamp_start = tm.time()
                # 视频播放结束之后average_loss和average_q才会有值，表示整个视频的平均loss和q



                reward_frame, bit_rate, target_buffer, latency_limit,action,average_loss,average_q = self.abr.run_frame(time, time_interval, send_data_size, chunk_len, \
                rebuf, buffer_size, play_time_len, end_delay, \
                cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
                buffer_flag, cdn_flag, skip_flag, end_of_video,action,last_action,frame_time_len)
                last_action = action
                cnt += 1
                frame_all_sum += 1
                timestamp_end = tm.time()
                call_time_sum += timestamp_end - timestamp_start
                # reward_all表示整个视频的reward，即episode的reward
                reward_all += reward_frame
                if end_of_video:
                    print("network traceID %d, network_reward %.3f, avg_running_time %.5f,average_loss %.3f,average_q %.3f" %
                          (trace_count, reward_all, call_time_sum/cnt,average_loss,average_q))
                    # reward_all_sum表示整个epoch的reward总和
                    reward_all_sum += reward_all
                    # loss_all_sum += average_loss
                    # q_all_sum += average_q
                    run_time += call_time_sum / cnt
                    self.episode_count += 1
                    cnt = 0
                    break
                #fixme 如果视频结束是否需要清空buffer
            self._update_target_net(False)
            self._save_net()
            trace_count += 1
            if trace_count >= len(self.all_file_names):
                print("\n %s EPOCH FINISH %d,average_step=%d,average_reward=%.2f,average_runtime=%.5f \n\n" %
                      (tm.strftime(TIME_FORMAT),
                       i,
                       frame_all_sum // trace_count,
                       reward_all_sum / trace_count,
                       run_time / trace_count))
                break
        return

    def _update_target_net(self,random_action=False):
        # print(random_action)
        if not self.testing and self.episode_count == self.update_target_episode and not random_action:
            self.target_net_update_count += 1
            print("%s UPDATE TARGET NET,interval %.3f,update count %d\n" % (tm.strftime(TIME_FORMAT),self.update_target_interval,self.update_target_interval))
            self.update_target_episode = int(self.update_target_episode + self.update_target_interval)
            self.update_target_interval = min((self.update_target_interval + UPDATE_TARGET_RATE),UPDATE_TARGET_BY_EPISODE_END)
            self.abr.update_target_net()
        return

    def _save_net(self):
        if not self.testing:
            self.abr.save_params_to_file(model_path=MODEL_PATH,mark='dqn')
            # self.abr.sa


    def run(self):
        for i in range(1, EPOCH_NUM):
            self._run_epoch(i)


if __name__ == '__main__':
    runner = Experiment(False)
    runner.run()