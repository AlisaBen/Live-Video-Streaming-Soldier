# Author: Taoz
# Date  : 8/29/2018
# Time  : 3:08 PM
# FileName: config.py


import time
import sys
import configparser
import os


def load_conf(conf_file):
    # 获取当前文件路径
    current_path = os.path.abspath(__file__)
    # config.ini文件路径,获取当前目录的父目录的父目录与congig.ini拼接
    default_conf_file = os.path.join(os.path.abspath(os.path.dirname(current_path)),
                                     'dqn_default_conf.ini')
    # print('default conf file:', default_conf_file)
    #
    # print('customer_conf_file:', conf_file)

    # print(os.path.exists(default_conf_file))
    # print(os.path.exists(conf_file))

    config = configparser.ConfigParser(allow_no_value=True, interpolation=configparser.ExtendedInterpolation())

    # config.read(default_conf_file)
    config.read(conf_file)
    config.sections()
    return config['DQN']

# sys.argv[1] = ".\configurations\dqn_conf_1001.ini"
# print(sys.argv[1])

customer_conf_file = sys.argv[1]

dqn_conf = load_conf(customer_conf_file)

"""experiment"""
GPU_INDEX = dqn_conf.getint('GPU_INDEX')
PRE_TRAIN_MODEL_FILE = dqn_conf.get('PRE_TRAIN_MODEL_FILE')
EPOCH_NUM = dqn_conf.getint('EPOCH_NUM')
EPOCH_LENGTH = dqn_conf.getint('EPOCH_LENGTH')
RANDOM_SEED = int(time.time() * 1000) % 100000000

ACTION_NUM = dqn_conf.getint('ACTION_NUM')
# OBSERVATION_TYPE = dqn_conf.get('OBSERVATION_TYPE')
# CHANNEL = dqn_conf.getint('CHANNEL')
# WIDTH = dqn_conf.getint('WIDTH')
# HEIGHT = dqn_conf.getint('HEIGHT')

FRAME_SKIP = dqn_conf.getint('FRAME_SKIP')
TEST = dqn_conf.getboolean('TEST')
TIME_FORMAT = dqn_conf.get('TIME_FORMAT')
"""player"""
TRAIN_PER_STEP = dqn_conf.getint('TRAIN_PER_STEP')

"""replay buffer"""
PHI_LENGTH = dqn_conf.getint('PHI_LENGTH')
if TEST:
    BUFFER_MAX = 50
else:
    BUFFER_MAX = dqn_conf.getint('BUFFER_MAX')
BEGIN_RANDOM_STEP = dqn_conf.getint('BEGIN_RANDOM_STEP')

"""q-learning"""
DISCOUNT = dqn_conf.getfloat('DISCOUNT')
EPSILON_MIN = dqn_conf.getfloat('EPSILON_MIN')
EPSILON_START = dqn_conf.getfloat('EPSILON_START')
EPSILON_DECAY = dqn_conf.getint('EPSILON_DECAY')

IS_DOUBLE = dqn_conf.getboolean('IS_DOUBLE')
IS_DUELING = dqn_conf.getboolean('IS_DUELING')
IS_NOISY = dqn_conf.getboolean('IS_NOISY')
#
# """noisy-net option"""
# NOISY_SCALE = dqn_conf.getfloat('NOISY_SCALE')
# NOISY_ALPHA = dqn_conf.getfloat('NOISY_ALPHA')
#
UPDATE_TARGET_BY_EPISODE_END = dqn_conf.getint('UPDATE_TARGET_BY_EPISODE_END')
UPDATE_TARGET_BY_EPISODE_START = dqn_conf.getint('UPDATE_TARGET_BY_EPISODE_START')
UPDATE_TARGET_DECAY = dqn_conf.getint('UPDATE_TARGET_DECAY')
UPDATE_TARGET_RATE = (UPDATE_TARGET_BY_EPISODE_END - UPDATE_TARGET_BY_EPISODE_START) / UPDATE_TARGET_DECAY + 0.000001

N_STEP = dqn_conf.getint('N_STEP')

OPTIMIZER = dqn_conf.get('OPTIMIZER')
LEARNING_RATE = dqn_conf.getfloat('LEARNING_RATE')
WEIGHT_DECAY = dqn_conf.getfloat('WEIGHT_DECAY')
GRAD_CLIPPING_THETA = dqn_conf.getfloat('GRAD_CLIPPING_THETA')

POSITIVE_REWARD = dqn_conf.getfloat('POSITIVE_REWARD')
NEGATIVE_REWARD = dqn_conf.getfloat('NEGATIVE_REWARD')
# LIVING_REWARD = dqn_conf.getfloat('LIVING_REWARD')
# SPECIAL_PUNISH = dqn_conf.getfloat('SPECIAL_PUNISH')
#
# FOOD_REWARD = dqn_conf.getfloat('FOOD_REWARD')
# KILL_REWARD = dqn_conf.getfloat('KILL_REWARD')
# """OTHER"""
MODEL_FILE_MARK = dqn_conf.get('MODEL_FILE_MARK')
BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")
# if TEST:
#     MODEL_PATH = "/home/benyafang/workspace/server/medusa/results/filter_model_channel_" + dqn_conf.get('CHANNEL')
# else:
#     MODEL_PATH = "/home/benyafang/workspace/server/medusa/results/" + MODEL_FILE_MARK
# MODEL_PATH = "/home/benyafang/workspace/server/medusa/results/" + MODEL_FILE_MARK
MODEL_PATH = dqn_conf.get('MODEL_PATH')
#
EDITED_TIME = dqn_conf.get("EDITED_TIME")

SMOOTH_PENALTY= dqn_conf.getfloat("SMOOTH_PENALTY")
LANTENCY_PENALTY = dqn_conf.getfloat("LANTENCY_PENALTY")
REBUF_PENALTY = dqn_conf.getfloat("REBUF_PENALTY")
SKIP_PENALTY = dqn_conf.getfloat("SKIP_PENALTY")

# print('\n\n\n\n++++++++++++++++ config edited time: %s ++++++++++++++++++' % EDITED_TIME)
# print('BEGIN_TIME:', BEGIN_TIME)
# print('CONF FILE:', customer_conf_file)
# print('GAME_NAME:', GAME_NAME)
# print('--------------------------')

print('configuration:')
for k, v in dqn_conf.items():
    print('[%s = %s]' % (k, v))
# print('MODEL_PATH:', MODEL_PATH)
print('--------------------------')
