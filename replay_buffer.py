import numpy as np
class ReplayBuffer():
    def __init__(self,buffer_size=7500, past_frame_len=10,multi_step=0,gamma=0.99):
        # super(ReplayBuffer,self).__init__(buffer_size, past_frame_len)
        self.past_frame_len = past_frame_len
        self.buffer_size = buffer_size
        self.multi_step = multi_step
        self.gamma = gamma
        self.rng = np.random.RandomState(40)
        self.S_time_interval = [0] * buffer_size  # 时间间隔
        self.S_send_data_size = [0] * buffer_size  # 发送数据大小
        self.S_chunk_len = [0] * buffer_size  # 块长度
        self.S_rebuf = [0] * buffer_size  # 重新缓冲，拒绝，保证较少的拒绝事件
        self.S_buffer_size = [0] * buffer_size  # 缓冲大小
        self.S_end_delay = [0] * buffer_size  # 结束延迟
        self.S_chunk_size = [0] * buffer_size  #
        self.S_play_time_len = [0] * buffer_size  # 播放时间长度
        self.S_decision_flag = [0] * buffer_size  # 决定的标志
        self.S_buffer_flag = [0] * buffer_size  # buffer的标志
        self.S_cdn_flag = [0] * buffer_size  # cdn的标志
        self.S_skip_time = [0] * buffer_size  # 跳帧的时间
        self.S_end_of_video = [0] * buffer_size
        self.rewards = [0] * buffer_size
        self.actions = [0] * buffer_size
        return

    def insert_sample(self,time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len, end_delay, cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, buffer_flag, cdn_flag, skip_flag, end_of_video,reward,action):
        self.S_time_interval.pop(0)
        self.S_send_data_size.pop(0)
        self.S_chunk_len.pop(0)
        self.S_buffer_size.pop(0)
        self.S_rebuf.pop(0)
        self.S_end_delay.pop(0)
        self.S_play_time_len.pop(0)
        self.S_decision_flag.pop(0)
        self.S_buffer_flag.pop(0)
        self.S_cdn_flag.pop(0)
        self.S_skip_time.pop(0)
        self.S_end_of_video.pop(0)
        self.rewards.pop(0)
        self.actions.pop(0)

        self.S_time_interval.append(time_interval)
        self.S_send_data_size.append(send_data_size)
        self.S_chunk_len.append(chunk_len)
        self.S_buffer_size.append(buffer_size)
        self.S_rebuf.append(rebuf)  # 拒绝
        self.S_end_delay.append(end_delay)
        self.S_play_time_len.append(play_time_len)
        self.S_decision_flag.append(decision_flag)
        self.S_buffer_flag.append(buffer_flag)
        self.S_cdn_flag.append(cdn_flag)
        self.S_skip_time.append(skip_frame_time_len)
        self.S_end_of_video.append(end_of_video)
        self.rewards.append(reward)
        self.actions.append(action)
        return

    def get_current_state(self):
        st = []
        st.append(self.S_time_interval[-self.past_frame_len:])
        st.append(self.S_send_data_size[-self.past_frame_len:])
        st.append(self.S_chunk_len[-self.past_frame_len:])
        st.append(self.S_buffer_size[-self.past_frame_len:])
        st.append(self.S_rebuf[-self.past_frame_len:])
        st.append(self.S_end_delay[-self.past_frame_len:])
        st.append(self.S_play_time_len[-self.past_frame_len:])
        st.append(self.S_decision_flag[-self.past_frame_len:])
        st.append(self.S_cdn_flag[-self.past_frame_len:])
        st.append(self.S_skip_time[-self.past_frame_len:])
        st.append(self.S_end_of_video[-self.past_frame_len:])
        return np.array(st).transpose((1,0))

    def get_certain_state(self,index):
        st = []
        st.append(self.S_time_interval[index:index + self.past_frame_len])
        st.append(self.S_send_data_size[index:index + self.past_frame_len])
        st.append(self.S_chunk_len[index:index + self.past_frame_len])
        st.append(self.S_buffer_size[index:index + self.past_frame_len])
        st.append(self.S_rebuf[index:index + self.past_frame_len])
        st.append(self.S_end_delay[index:index + self.past_frame_len])
        st.append(self.S_play_time_len[index:index + self.past_frame_len])
        st.append(self.S_decision_flag[index:index + self.past_frame_len])
        st.append(self.S_cdn_flag[index:index + self.past_frame_len])
        st.append(self.S_skip_time[index:index + self.past_frame_len])
        st.append(self.S_end_of_video[index:index + self.past_frame_len])
        return np.array(st).transpose((1,0))

    def get_batch(self,batch_size):
        state = np.zeros((batch_size,self.past_frame_len,11))
        s_ = np.zeros((batch_size,self.past_frame_len,11))
        actions = np.zeros((batch_size,1))
        rewards = np.zeros((batch_size,1))
        cnt = 0
        while cnt < batch_size:
            index = self.rng.randint(0,self.buffer_size - self.past_frame_len - self.multi_step + 1)
            # all_index = np.arange(index,index + self.past_frame_len)
            # next_st_index = np.arange(index + self.multi_step, index + self.multi_step + self.past_frame_len)
            end_index = index + self.past_frame_len - 1
            state[cnt] = self.get_certain_state(index)
            s_[cnt] = self.get_certain_state(index + self.multi_step)
            actions[cnt] = self.actions[end_index]
            rewards[cnt] = self.rewards[end_index]

            for i in range(1, self.multi_step):
                rewards[cnt] += self.rewards[end_index + i] * self.gamma ** i
            cnt += 1
        return state,s_,actions,rewards

if __name__ == '__main__':
    a = [1,22,3,4,5,6]
    # d = a[0]
    # c = [3,4,5,6,78,8]
    # b = [a[-2:],c[-2:]]
    for i in range(1,10):
        print(i)
    replay_buffer = ReplayBuffer(100,5)
    cnt = 1
    while cnt < 110:
        replay_buffer.insert_sample(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
        cnt += 1
    st = replay_buffer.get_current_state()
    st_ = replay_buffer.get_certain_state(3)
    state, s_, actions, rewards = replay_buffer.get_batch(16)
    print(st.shape)
    print(st_.shape)
    print(state.shape)
    print(actions.shape)
    # print(len(st)) # 11
    # print(len(st[0]))  # 5