"""
desc：
    DKT+模型源代码
info:
    模型主函数
"""
import os
import tensorflow as tf
import time
import numpy as np

from utils import DKT
from load_data import DKTData

import argparse

"""
Assignable variables:
num_runs: int
num_epochs: int
keep_prob: float
batch_size: int
hidden_layer_structure: tuple
data_dir: str
train_file_name: str
test_file_name: str
ckpt_save_dir: str
"""

# 创建一个解析对象
parser = argparse.ArgumentParser()

# 网络配置
# network configuration
# 向该解析对象添加相关命令行参数和选项
parser.add_argument("-hl", "--hidden_layer_structure", default=[200, ], nargs='*', type=int,
                    help="The hidden layer structure in the RNN. If there is 2 hidden layers with first layer "
                         "of 200 and second layer of 50. Type in '-hl 200 50'")
parser.add_argument("-cell", "--rnn_cell", default='LSTM', choices=['LSTM', 'GRU', 'BasicRNN', 'LayerNormBasicLSTM'],
                    help='Specify the rnn cell used in the graph.')
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2,
                    help="The learning rate when training the model.")
parser.add_argument("-kp", "--keep_prob", type=float, default=0.5,
                    help="Keep probability when training the network.")
parser.add_argument("-mgn", "--max_grad_norm", type=float, default=5.0,
                    help="The maximum gradient norm allowed when clipping.")
parser.add_argument("-lw1", "--lambda_w1", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l1-norm.")
parser.add_argument("-lw2", "--lambda_w2", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l2-norm.")
parser.add_argument("-lo", "--lambda_o", type=float, default=0.00,
                    help="The lambda coefficient for the regularization objective.")

# 训练配置：总轮数，迭代次数，批量大小
# training configuration
parser.add_argument("--num_runs", type=int, default=5,
                    help="Number of runs to repeat the experiment.")
parser.add_argument("--num_epochs", type=int, default=500,
                    help="Maximum number of epochs to train the network.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="The mini-batch size used when training the network.")

# 数据文件配置：训练集与测试集数据
# data file configuration
parser.add_argument('--data_dir', type=str, default='./data/',
                    help="the data directory, default as './data/")
parser.add_argument('--train_file', type=str, default='skill_id_train.csv',
                    help="train data file, default as 'skill_id_train.csv'.")
parser.add_argument('--test_file', type=str, default='skill_id_test.csv',
                    help="train data file, default as 'skill_id_test.csv'.")
parser.add_argument("-csd", "--ckpt_save_dir", type=str, default=None,
                    help="checkpoint save directory")
parser.add_argument('--dataset', type=str, default='a2009')

# 从对象中返回参数值
args = parser.parse_args()

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

# 选择数据集，默认是a2009数据集，对应skill_id_train文件
dataset = args.dataset

if dataset == 'a2009u':
    train_path = './data/assist2009_updated/assist2009_updated_train.csv'
    test_path = './data/assist2009_updated/assist2009_updated_test.csv'
    save_dir_prefix = './a2009u/'
elif dataset == 'a2015':
    train_path = './data/assist2015/assist2015_train.csv'
    test_path = './data/assist2015/assist2015_test.csv'
    save_dir_prefix = './a2015/'
elif dataset == 'synthetic':
    train_path = './data/synthetic/naive_c5_q50_s4000_v1_train.csv'
    test_path = './data/synthetic/naive_c5_q50_s4000_v1_test.csv'
    save_dir_prefix = './synthetic/'
elif dataset == 'statics':
    train_path = './data/STATICS/STATICS_train.csv'
    test_path = './data/STATICS/STATICS_test.csv'
    save_dir_prefix = './STATICS/'
elif dataset == 'assistment_challenge':
    train_path = './data/assistment_challenge/assistment_challenge_train.csv'
    test_path = './data/assistment_challenge/assistment_challenge_test.csv'
    save_dir_prefix = './assistment_challenge/'
elif dataset == 'toy':
    train_path = './data/toy_data_train.csv'
    test_path = './data/toy_data_test.csv'
    save_dir_prefix = './toy/'
elif dataset == 'a2009':
    train_path = './data/skill_id_train.csv'
    test_path = './data/skill_id_test.csv'
    save_dir_prefix = './a2009/'

# 网络配置
network_config = {'batch_size': args.batch_size,
                  'hidden_layer_structure': list(args.hidden_layer_structure),
                  'learning_rate': args.learning_rate,
                  'keep_prob': args.keep_prob,
                  'rnn_cell': rnn_cells[args.rnn_cell],
                  'lambda_w1': args.lambda_w1,
                  'lambda_w2': args.lambda_w2,
                  'lambda_o': args.lambda_o}

num_runs = args.num_runs
num_epochs = args.num_epochs
batch_size = args.batch_size
keep_prob = args.keep_prob

# 检查点保存路径
ckpt_save_dir = args.ckpt_save_dir


# 定义主函数
def main():
    # 配置tf.Session的运算方式，比如gpu运算或者cpu运算
    config = tf.ConfigProto()
    # 当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("模型开始训练！")

    # 从文件中读取数据（核心步骤一）
    data = DKTData(train_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_test = data.test
    num_problems = data.num_problems

    # 创建DKT模型
    dkt = DKT(sess, data_train, data_test, num_problems, network_config,
              save_dir_prefix=save_dir_prefix,
              num_runs=num_runs, num_epochs=num_epochs,
              keep_prob=keep_prob, logging=True, save=True)

    # 建立计算图（核心步骤二）
    # run optimization of the created model
    dkt.model.build_graph()
    # 运行模型（核心步骤三）
    dkt.run_optimization()
    # 关闭会话
    # close the session
    sess.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("模型运行时间 program run for: {:.2f}min".format((end_time - start_time)/60))
