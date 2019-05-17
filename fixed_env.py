import numpy as np
import random
import datetime

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42

VIDEO_CHUNCK_LEN = 2000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 4  # 比特等级
BUFFER_LEVELS = 2  # 缓冲可选的等级
CHUNK_TIME_LEN = 2  # 块的时长
Target_buffer = [2.0, 3.0]

lamda = 1
default_quality = 0
#latency_threshold = 3
#skip_add_frame = 100

FPS = 25.0


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED,
                 logfile_path='./', VIDEO_SIZE_FILE='./video_size_',
                 Debug=True):
        assert len(all_cooked_time) == len(all_cooked_bw)

        if Debug:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.log_file = open(logfile_path + "log." + current_time, "w")

        self.video_size_file = VIDEO_SIZE_FILE
        self.Debug = Debug

        # np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.time = 0
        self.play_time = 0
        self.play_time_counter = 0
        self.newest_frame = 0  # 最新的帧号
        self.next_I_frame = 50  # 下一个I帧

        self.video_chunk_counter = 0  # 视频块的数量
        self.buffer_size = 0  # 缓冲区的大小表示缓冲区目前能够支持的播放时长

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        # randomize the start point of the trace
        # note: trace file starts with time 0
        # self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        # self.mahimahi_ptr = 1
        self.decision = False  # 是否是I帧
        self.buffer_status = True  # buffer的状态，True表示正在缓冲
        self.skip_time_frame = 1000000000  # 跳帧时间
        self.add_frame = 0  # 增加帧
        self.skip_to_frame = self.skip_time_frame  # 跳到的帧号

        # self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size = {}  # in bytes，转换成字节
        self.cdn_arrive_time = {}  # CDN到达时间
        # gop是一个画面组
        self.gop_time_len = {}
        self.gop_flag = {}  # GOP的标志位，标志视频块的I帧和P帧
        """读取视频文件中的数据存储到内存中"""
        for bitrate in range(BITRATE_LEVELS):
            # 遍历每一个码率等级
            self.video_size[bitrate] = []  # 每一个码率对应的视频大小存储一个列表
            self.cdn_arrive_time[bitrate] = []  # 每一个码率对应的CDN到达的时间存储一个列表
            self.gop_time_len[bitrate] = []  # 每一个码率存储一个画面组的时长一个列表
            self.gop_flag[bitrate] = []  # 每一个码率存储一个GOP的标志位列表
            cnt = 0  # count
            with open(self.video_size_file + str(bitrate)) as f:
                # 读取不同码率下的视频文件的信息
                for line in f:
                    # print(line.split(), bitrate)
                    self.video_size[bitrate].append(float(line.split()[1]))  # 文件中的第二个数据表示视频块的大小
                    self.gop_time_len[bitrate].append(float(1/FPS))  # 画面组的时长1/FPS
                    self.gop_flag[bitrate].append(int(float(line.split()[2])))  # 文件中的第三个数据表示gop的标志位
                    self.cdn_arrive_time[bitrate].append(float(line.split()[0]))  # 文件中的第一个数据表示到达CDN的时间
                    cnt += 1
        self.gop_remain = self.video_size[default_quality][0]  # remain的画面组是最小比特中的第一个视频段的视频大小
        self.latency = self.gop_time_len[0][0]  # 延迟是最小比特中的第一个视频块的gop时长

    def get_trace_id(self):
        return self.trace_idx

    def get_video_frame(self, quality, target_buffer, latency_limit):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        # Initial the Settings
        self.decision = False 
        self.add_frame = 0                                                    # GOP_flag
        video_frame_size = self.video_size[quality][self.video_chunk_counter] # Data_size，该码率对应下的正在播放的视频块的大小
        cdn_rebuf_time = 0                                                    # CDN_rebuf_timeCDN拒绝时间
        rebuf = 0                                                             # rebuf_time拒绝时间
        FRAME_TIME_LEN = float(1/FPS)                                         # frame time len每帧的时长
        end_of_video = False                                                  # is trace time end
        duration = 0                                                          # this loop 's time len
        # duration表示循环的时长
        current_new = 0
        self.skip = False
        latency_threshold = latency_limit  # 延迟的阈值
        # 快播和慢播
        if target_buffer == 0: # 缓冲区大小是正常的一半大小，快播减半，慢播少减半
            quick_play_bound = 1.0
            slow_play_bound = 0.3
        else: # 缓冲区的大小分为正常大小和一半大小，如果缓冲区是正常大小，快播就是2倍速，慢播0.5倍速
            quick_play_bound = 2.0
            slow_play_bound = 0.5
        # This code is check the quick play or slow play
        # output is the play_weight
        # 如果缓冲的大小大于快播的跳跃，播放权重为1.05
        # 如果缓冲的大小小于慢波的跳跃，播放权重为0.95
        # 否则的化，播放权重为1
        if self.buffer_size > quick_play_bound:
            # quick play
            play_duration_weight = 1.05
            # if Debug:
                # print("kuaibo\n")
            # elif self.buffer_size < slow_play_bound:
        elif self.buffer_size < slow_play_bound:
            # slow play
            play_duration_weight = 0.95
            # if Debug:
                # print("manbo\n")
        else:
            play_duration_weight = 1

        # This code is check Is the cdn has the frame in this time
        # self.time means the real time
        # self.cdn_arrive_time means the time the frame came
        """
        # CDN到达时间表示这一帧到达的时间
        # 该段代码检测当前CDN中是否有该帧的视频
        """
        if self.time < self.cdn_arrive_time[quality][self.video_chunk_counter] and not end_of_video: # CDN can't get the frame
            # 现在的时间还没有到该视频块预计到达CDN的时间，所以CDN服务器中没有该视频
            # CDN到达的时间减去现在的时间是CDN接受到数据的拒绝的时间长度
            cdn_rebuf_time = self.cdn_arrive_time[quality][self.video_chunk_counter] - self.time
            self.newest_frame = self.video_chunk_counter
            duration = cdn_rebuf_time
            # if the client means the buffering
            if not self.buffer_status:
                # not buffering ,you can't get the frame ,but you must play the frame
                # the buffer is enough
                """
                # 缓冲区是满的，CDN拒绝的时间长度乘以播放速率，如果缓冲区的大小比乘积大，
                # 剩余的缓冲区的大小等于当前减去预计播放的大小
                播放的时长加上预计的播放的时长
                拒绝的时长更新为0
                
                缓冲区不满，播放的时间应该要加上缓冲区的大小，
                拒绝的时间为CDN拒绝的时间减去常速播放下的缓冲区大小除以播放速率
                缓冲区大小更新为0
                buffer的状态为True，表示没有视频可播放
                """
                if self.buffer_size > cdn_rebuf_time * play_duration_weight:
                    self.buffer_size -= cdn_rebuf_time * play_duration_weight
                    self.play_time += cdn_rebuf_time * play_duration_weight
                    rebuf = 0
                    play_len = cdn_rebuf_time * play_duration_weight
                # not enough .let the client buffering
                else:
                    self.play_time += self.buffer_size
                    rebuf = cdn_rebuf_time - (self.buffer_size / play_duration_weight)
                    play_len = self.buffer_size
                    self.buffer_size = 0
                    self.buffer_status = True

            # calculate the play time , the real time ,the latency
                # the normal play the frame
                """
                如果播放时间的counter大于等于跳帧的大小，播放视频的块的index就是应该跳到的那一帧
                    播放时间就是跳帧视频块乘以帧时长，该视频总共的播放时长？？
                否则，播放的视频快的index为播放时长除以帧时长取整？？？？？？？
                
                """
                if self.play_time_counter >= self.skip_time_frame:
                    self.play_time_counter = self.skip_to_frame
                    self.play_time = self.play_time_counter * FRAME_TIME_LEN
                    if self.Debug:
                        self.log_file.write("ADD_Frame" + str(self.add_frame) + "\n")
                    
                    self.add_frame = 0
                else:
                    self.play_time_counter = int(self.play_time/FRAME_TIME_LEN)
                """延迟等于最新帧号减去视频块counter乘以帧时长加上缓冲区大小"""
                """当前时间更新为CDN到达的时间"""
                self.latency = (self.newest_frame - self.video_chunk_counter) * FRAME_TIME_LEN  + self.buffer_size
                self.time = self.cdn_arrive_time[quality][self.video_chunk_counter]
            else:
                """正在缓冲"""
                rebuf = duration
                play_len = 0
                self.time = self.cdn_arrive_time[quality][self.video_chunk_counter]
                self.latency = (self.newest_frame - self.video_chunk_counter) * FRAME_TIME_LEN  + self.buffer_size
            # Debug info
            '''print("%.4f"%self.time ,
                  "  cdn %4f"%cdn_rebuf_time,
                  "~rebuf~~ %.3f"%rebuf,
                  "~dur~~%.4f"%duration,
                  "~delay~%.4f"%(cdn_rebuf_time),
                  "~id! ", self.video_chunk_counter,
                  "~newest ", self.newest_frame,
                  "~~buffer~~ %4f"%self.buffer_size,
                  "~~play_time~~%4f"%self.play_time ,
                  "~~play_id",self.play_time_counter,
                  "~~latency~~%4f"%self.latency,"000")'''
            if self.Debug:
                self.log_file.write("real_time %.4f\t" % self.time +
                                    "cdn_rebuf%.4f\t" % cdn_rebuf_time +
                                    "client_rebuf %.3f\t" % rebuf +
                                    "download_duration %.4f\t" % duration +
                                    "frame_size %.4f\t" % video_frame_size +
                                    "play_time_len %.4f\t" % (play_len) +
                                    "download_id %d\t" % (self.video_chunk_counter-1) +
                                    "cdn_newest_frame %d\t" % self.newest_frame +
                                    "client_buffer %.4f\t" % self.buffer_size +
                                    "play_time %.4f\t" % self.play_time +
                                    "play_id %.4f\t" % self.play_time_counter +
                                    "latency %.4f\t" % self.latency + "000\n")
            # Return the loop
            cdn_has_frame = []
            for bitrate in range(BITRATE_LEVELS):
                """每一个码率对应的当前视频块到最新帧号之间的视频大小"""
                cdn_has_frame_temp = self.video_size[bitrate][self.video_chunk_counter : self.newest_frame]
                cdn_has_frame.append(cdn_has_frame_temp)
            # GOP的标志位
            cdn_has_frame.append(self.gop_flag[bitrate][self.video_chunk_counter:self.newest_frame])

            return  [self.time,                       # physical time 物理时间
                    duration,                         # this loop duration, means the download time # 下载时间
                    0,                                # frame data size  # 帧数据大小
                    0,                                # frame time len  # 帧时长
                    rebuf,                            # rebuf len  # 拒绝时间
                    self.buffer_size,                 # client buffer  # 缓冲区大小
                    play_len,                         # play time len  # 播放时长
                    self.latency ,                    # latency  # 延迟时间
                    self.newest_frame,                # cdn the newest frame id  #CDN最新帧的ID
                    (self.video_chunk_counter - 1),   # download_id  # 下载的ID
                    cdn_has_frame,                    # CDN_has_frame  # CDN有帧
                    self.add_frame * FRAME_TIME_LEN,       # the num of skip frame 
                    self.decision,                    # Is_I frame edge  # 是否是I帧
                    self.buffer_status,               # Is the buffer is buffering  # 缓冲区是否正在缓冲
                    True,                             # Is the CDN has no frame  # CDN没有帧
                    self.skip,                        # Is the events of skip frame  # 跳帧时间
                    end_of_video]                     # Is the end of video  # 是否是视频的最后一帧
        else:
            """CDN能够获得该帧，或者视频的结束"""
            the_newst_frame = self.video_chunk_counter
            current_new = self.cdn_arrive_time[quality][the_newst_frame]
            # 当前CDN到达时间小于物理时间
            while (current_new < self.time):
                # 找到当前时间的前一帧的视频，得到CDN的到达时间
                the_newst_frame += 1
                current_new = self.cdn_arrive_time[quality][the_newst_frame]
            self.newest_frame = the_newst_frame
        # If the CDN can get the frame:
        """0.5s采样一次，物理时间除以采样频率得到的采样的个数，如果这个个数大于了实际采样个数，视频播放结束"""
        if int(self.time / 0.5) >= len(self.cooked_bw):
            end_of_video = True
        else:
            throughput = self.cooked_bw[int(self.time / 0.5)] * B_IN_MB  # 转换成字节带宽单位
            #rtt        = self.cooked_rtt[int(self.time / 0.5)]
            duration = float(video_frame_size / throughput)  # 视频大小除以带宽得到播放视频的时长
        # If the the frame is the Gop end ,next will be the next I frame
        """下一个画面组是否是I帧，以及下一个I帧的帧号"""
        if self.gop_flag[quality][self.video_chunk_counter + 1] == 1:
            self.decision = True
            self.next_I_frame = self.video_chunk_counter + 50 + 1
        # If the buffering
        if self.buffer_status and not end_of_video:
            # let the buffer_size to expand to the target_buffer
            if self.buffer_size < Target_buffer[target_buffer]:
                """扩展缓冲大小到目标缓冲大小"""
                rebuf = duration
                # 新的缓冲大小加上一个gop画面组的时长，当前时间加上duration
                self.buffer_size += self.gop_time_len[quality][self.video_chunk_counter]
                self.time += duration
            # if the buffer is enough
            else:
                """缓冲区大小比目标缓冲区大，不再缓冲"""
                self.buffer_status = False
                rebuf = duration
            # calculate the play time , the real time ,the latency
            self.play_time_counter = int(self.play_time/FRAME_TIME_LEN)  # 播放时长除以帧速率
            self.latency = (self.newest_frame - self.video_chunk_counter) * FRAME_TIME_LEN + self.buffer_size
            # Debug Info
            """延迟时间超过了延迟阈值"""
            if self.latency > latency_threshold :
                self.skip_time_frame = self.video_chunk_counter
                if self.newest_frame >= self.next_I_frame:
                    """最新帧号超过了下一个I帧"""
                    self.add_frame = self.next_I_frame - self.skip_time_frame -1 
                    self.video_chunk_counter = self.next_I_frame
                    self.skip_to_frame = self.video_chunk_counter 
                    self.next_I_frame += 50
                    self.latency = (self.newest_frame - self.video_chunk_counter) * FRAME_TIME_LEN + self.buffer_size
                    self.skip = True
                    self.decision = True
                    
                else:
                    self.add_frame = 0
                    self.video_chunk_counter += 1
                    self.skip_to_frame = self.video_chunk_counter

                if self.Debug:
                    self.log_file.write("skip events: skip_time_frame, play_frame, new_download_frame, ADD_frame" + str(self.skip_time_frame) + " " + str(self.play_time_counter) + " " + str(self.video_chunk_counter) +" " +str(self.add_frame) + "\n")
            else:
                self.video_chunk_counter += 1
            '''print("%.4f"%self.time ,
                      "  cdn %4f"%cdn_rebuf_time,
                      "~rebuf~~ %.3f"%rebuf,
                      "~dur~~%.4f"%duration,
                      "~delay~%.4f"%(cdn_rebuf_time),
                      "~id! ", self.video_chunk_counter,
                      "~newest ", self.newest_frame,
                      "~~buffer~~ %4f"%self.buffer_size,
                      "~~play_time~~%4f"%self.play_time ,
                      "~~play_id",self.play_time_counter,
                      "~~latency~~%4f"%self.latency,"111")'''
            if self.Debug:
                 self.log_file.write("real_time %.4f\t" % self.time +
                                      "cdn_rebuf%.4f\t" % cdn_rebuf_time +
                                      "client_rebuf %.3f\t" % rebuf  +
                                      "download_duration %.4f\t" % duration +
                                      "frame_size %.4f\t" % video_frame_size +
                                      "play_time len %.4f\t" % 0 +
                                      "download_id %d\t" % (self.video_chunk_counter-1) +
                                      "cdn_newest_frame %d\t" % self.newest_frame +
                                      "client_buffer %.4f\t" % self.buffer_size +
                                      "play_time %.4f\t" % self.play_time +
                                      "play_id %.4f\t" % self.play_time_counter +
                                      "latency %.4f\t" % self.latency + "111\n")
            cdn_has_frame = []
            for bitrate in range(BITRATE_LEVELS):
                cdn_has_frame_temp = self.video_size[bitrate][self.video_chunk_counter : self.newest_frame]
                cdn_has_frame.append(cdn_has_frame_temp)
            cdn_has_frame.append(self.gop_flag[bitrate][self.video_chunk_counter:self.newest_frame])
            # Return the loop
            
            return [self.time,                      # physical time
                    duration,                        # this loop duration, means the download time
                    video_frame_size,                # frame data size
                    FRAME_TIME_LEN,                  # frame time len
                    rebuf,                           # rebuf len
                    self.buffer_size,                # client buffer
                    0,                               # play time len
                    self.latency ,                   # latency
                    self.newest_frame,               # cdn the newest frame id
                    (self.video_chunk_counter - 1),  # download_id
                    cdn_has_frame,                   # CDN_has_frame
                    self.add_frame * FRAME_TIME_LEN,      # the num of skip frame
                    self.decision,                   # Is_I frame edge
                    self.buffer_status,              # Is the buffer is buffering
                    False,                           # Is the CDN has no frame
                    self.skip,                      # Is the events of skip frame
                    end_of_video]                    # Is the end of video
        # If not buffering
        elif not end_of_video:
            # the normal loop
            # the buffer is enough
            if self.buffer_size > duration * play_duration_weight:
                self.buffer_size -= duration * play_duration_weight
                self.play_time += duration * play_duration_weight
                rebuf = 0
            # the buffer not enough
            else:
                self.play_time += self.buffer_size
                rebuf = duration  - (self.buffer_size / play_duration_weight)
                self.buffer_size = 0
                self.buffer_status = True
            # the normal play the frame
            if  self.play_time_counter >= self.skip_time_frame :
                self.play_time_counter = self.skip_to_frame
                self.play_time = self.play_time_counter * FRAME_TIME_LEN
                if self.Debug:
                    self.log_file.write("ADD_Frame" + str(self.add_frame) + "\n")
                
                self.add_frame = 0
            else:
                self.play_time_counter = int(self.play_time/FRAME_TIME_LEN)
            self.latency = (self.newest_frame - self.video_chunk_counter) * FRAME_TIME_LEN + self.buffer_size
            #play add the time
            self.buffer_size += self.gop_time_len[quality][self.video_chunk_counter]
            self.time += duration
            if self.latency > latency_threshold:
                self.skip_time_frame = self.video_chunk_counter
                if self.newest_frame >= self.next_I_frame:
                    self.add_frame = self.next_I_frame - self.skip_time_frame -1 
                    self.video_chunk_counter = self.next_I_frame
                    self.skip_to_frame = self.video_chunk_counter
                    self.next_I_frame += 50 
                    self.latency = (self.newest_frame - self.video_chunk_counter) * FRAME_TIME_LEN + self.buffer_size
                    self.skip = True
                    self.decision = True
                else:
                    self.add_frame = 0
                    self.video_chunk_counter += 1
                    self.skip_to_frame = self.video_chunk_counter
                    
                if self.Debug:
                    self.log_file.write("skip events: skip_download_frame, play_frame, new_download_frame, ADD_frame" + str(self.skip_time_frame) + " " + str(self.play_time_counter) +" " + str(self.video_chunk_counter) +" " +str(self.add_frame) + "\n")
            else:
                self.video_chunk_counter += 1
            '''print("%.4f"%self.time ,
                      "  cdn %4f"%cdn_rebuf_time,
                      "~rebuf~~ %.3f"%rebuf,
                      "~dur~~%.4f"%duration,
                      "~delay~%.4f"%(cdn_rebuf_time),
                      "~id! ", self.video_chunk_counter,
                      "~newest ", self.newest_frame,
                      "~~buffer~~ %4f"%self.buffer_size,
                      "~~play_time~~%4f"%self.play_time ,
                      "~~play_id",self.play_time_counter,
                      "~~latency~~%4f"%self.latency,"222")'''
            if self.Debug:
                 self.log_file.write("real_time %.4f\t"%self.time +
                                      "cdn_rebuf%.4f\t"%cdn_rebuf_time +
                                      "client_rebuf %.3f\t"%rebuf  +
                                      "download_duration %.4f\t"%duration +
                                      "frame_size %.4f\t"%video_frame_size +
                                      "play_time_len %.4f\t"% (duration * play_duration_weight) +
                                      "download_id %d\t"%(self.video_chunk_counter-1) +
                                      "cdn_newest_frame %d\t"%self.newest_frame +
                                      "client_buffer %.4f\t"%self.buffer_size  +
                                      "play_time %.4f\t"%self.play_time +
                                      "play_id %.4f\t"%self.play_time_counter +
                                      "latency %.4f\t"%self.latency + "222\n")
            cdn_has_frame = []
            for bitrate in range(BITRATE_LEVELS):
                cdn_has_frame_temp = self.video_size[bitrate][self.video_chunk_counter : self.newest_frame]
                cdn_has_frame.append(cdn_has_frame_temp)
            cdn_has_frame.append(self.gop_flag[bitrate][self.video_chunk_counter:self.newest_frame])
            #return loop
            return      [self.time,                             # physical time
                        duration,                               # this loop duration, means the download time
                        video_frame_size,                       # frame data size
                        FRAME_TIME_LEN,                         # frame time len
                        rebuf,                                  # rebuf len
                        self.buffer_size,                       # client buffer
                        (duration * play_duration_weight),      # play time len
                        self.latency ,                          # latency
                        self.newest_frame,                      # cdn the newest frame id
                        (self.video_chunk_counter - 1),         # download_id
                        cdn_has_frame,                          # CDN_has_frame
                        self.add_frame * FRAME_TIME_LEN,             # the num of skip frame
                        self.decision,                          # Is_I frame edge
                        self.buffer_status,                     # Is the buffer is buffering
                        False,                                  # Is the CDN has no frame
                        self.skip,                              # Is the events of skip frame
                        end_of_video]                           # Is the end of video
        # if the video is end
        if  end_of_video:
            self.time = 0
            self.play_time = 0
            self.play_time_counter = 0
            self.newest_frame = 0

            self.video_chunk_counter = 0
            self.buffer_size = 0  # 缓冲区的大小表示缓冲区目前能够支持的播放时长

            # pick a random trace file
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0
            #self.trace_idx += 1
            #if self.trace_idx >= len(self.all_cooked_time):
            #    self.trace_idx = 0
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            self.decision = False
            self.buffer_status = True
            self.skip_time_frame = 1000000000
            self.next_I_frame = 50
            self.add_frame = 0
            self.skip_to_frame = self.skip_time_frame

            self.video_size = {}  # in bytes
            self.cdn_arrive_time = {}
            self.gop_time_len = {}
            self.gop_flag = {}
            for bitrate in range(BITRATE_LEVELS):
                self.video_size[bitrate] = []
                self.cdn_arrive_time[bitrate] = []
                self.gop_time_len[bitrate] = []
                self.gop_flag[bitrate] = []
                cnt = 0
                with open(self.video_size_file + str(bitrate)) as f:
                    for line in f:
                        self.video_size[bitrate].append(float(line.split()[1]))
                        self.gop_time_len[bitrate].append(float(1/FPS))
                        self.gop_flag[bitrate].append(int(float(line.split()[2])))
                        self.cdn_arrive_time[bitrate].append(float(line.split()[0]))
                        cnt += 1
            self.gop_remain = self.video_size[default_quality][0]
            self.latency = self.gop_time_len[0][0]
            cdn_has_frame = []
            for bitrate in range(BITRATE_LEVELS):
                cdn_has_frame_temp = self.video_size[bitrate][self.video_chunk_counter : self.newest_frame]
                cdn_has_frame.append(cdn_has_frame_temp)
            cdn_has_frame.append(self.gop_flag[bitrate][self.video_chunk_counter:self.newest_frame])
            
            return      [self.time,                             # physical time
                        duration,                               # this loop duration, means the download time
                        video_frame_size,                       # frame data size
                        FRAME_TIME_LEN,                         # frame time len
                        rebuf,                                  # rebuf len
                        self.buffer_size,                       # client buffer
                        (duration * play_duration_weight),      # play time len
                        self.latency ,                          # latency
                        self.newest_frame,                      # cdn the newest frame id
                        (self.video_chunk_counter - 1),         # download_id
                        cdn_has_frame,                          # CDN_has_frame
                        0,                                      #
                        self.decision,                          # Is_I frame edge
                        self.buffer_status,                     # Is the buffer is buffering
                        False,                                  # Is the CDN has no frame
                        False,                                  # Is the envents of skip frame
                        True]                                   # Is the end of video
