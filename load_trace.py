import os


COOKED_TRACE_FOLDER = './cooked_traces/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    print(cooked_files)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names

def load_trace_list(cooked_trace_folder_list):
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    file_path_list = []
    for folder in cooked_trace_folder_list:
        cooked_files = os.listdir(folder)
        for cooked_file in cooked_files:
            file_path = folder + cooked_file
            file_path_list.append(file_path)
    import random
    random.shuffle(file_path_list)
    # print(file_path_list)
    for file_path in file_path_list:
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(file_path)

    return all_cooked_time, all_cooked_bw, all_file_names


if __name__ == '__main__':
    NETWORK_TRACE = 'fixed'
    VIDEO_TRACE = 'AsianCup_China_Uzbekistan'
    network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
    video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
    all_cooked_time, all_cooked_bw, all_file_names = load_trace(network_trace_dir)
    print(len(all_cooked_bw))  # 20
    print(len(all_cooked_bw[0]))  # 5880
    print(len(all_file_names))  # 20
    # print(all_file_names)
    # print(all_cooked_time)
    # print(all_cooked_bw)
    network_trace = ['fixed','high','low','medium','middle']
    video = ['AsianCup_China_Uzbekistan','Fengtimo_2018_11_3']
    network_trace_dir_list = ['./dataset/network_trace/' + trace + '/' for trace in network_trace]
    video_trace_prefix_list = [ './dataset/video_trace/' + v + '/frame_trace_' for v in video]
    print(network_trace_dir_list)
    print(video_trace_prefix_list)
    all_cooked_time, all_cooked_bw, all_file_names = load_trace_list(network_trace_dir_list)
    print(len(all_cooked_bw))  # 20
    print(len(all_cooked_bw[0]))  # 5880
    print(len(all_file_names))  # 20



