"""
desc:
    DKT模型可视化实现
info:
    极为重要
"""
import os
import tensorflow as tf
import time
import numpy as np
from utils import DKT
from load_data import DKTData
import matplotlib.pyplot as plt

# 数据集选择
dataset = 'a2009'

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
elif dataset == 'a2009':
    train_path = './data/skill_id_train.csv'
    test_path = './data/skill_id_test.csv'
    save_dir_prefix = './'

# 网络模型搭建之一
# DKT模型
rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

# train_path = os.path.join('./data/', 'skill_id_train.csv')
# test_path = os.path.join('./data/', 'skill_id_test.csv')

network_config = {'batch_size': 32, 'hidden_layer_structure': [200], 'learning_rate': 0.01, 'keep_prob': 0.333,
                  'rnn_cell': rnn_cells["LSTM"], 'lambda_o': 0.0, 'lambda_w1': 0.0, 'lambda_w2': 0.0}

# save_dir_prefix = 'checkpoints/n200.lo0.1.lw10.03.lw230.0'

num_runs = 1
num_epochs = 1
batch_size = 32
keep_prob = 0.333

tf.reset_default_graph()

sess = tf.Session()

data = DKTData(train_path, test_path, batch_size=batch_size)
data_train = data.train
data_test = data.test
num_problems = data.num_problems

dkt_original = DKT(sess, data_train, data_test, num_problems, network_config,
                   num_runs=num_runs, num_epochs=num_epochs,
                   save_dir_prefix=save_dir_prefix,
                   keep_prob=keep_prob, logging=False, save=False)

# load the model
dkt_original.model.build_graph()
dkt_original.load_model()

# 调试错误发生点
m1_orig, m2_orig = dkt_original.consistency()
msg = '& 0.0 & 0.0 & 0.0 & {0:.5f} & {1:.5f} \\\\'.format(m1_orig, m2_orig)
print(dkt_original.waviness())
print(dkt_original.waviness_np())

# 网络模型搭建之二
# DKT+模型
rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

# train_path = os.path.join('./data/', 'skill_id_train.csv')
# test_path = os.path.join('./data/', 'skill_id_test.csv')

network_config = {'batch_size': 32, 'hidden_layer_structure': [200], 'learning_rate': 0.01, 'keep_prob': 0.333,
                  'rnn_cell': rnn_cells["LSTM"], 'lambda_o': 0.1, 'lambda_w1': 0.003, 'lambda_w2': 3.0}

# save_dir_prefix = 'checkpoints/n200.lo0.1.lw10.03.lw230.0'

num_runs = 1
num_epochs = 1
batch_size = 32
keep_prob = 0.333

tf.reset_default_graph()

sess = tf.Session()

data = DKTData(train_path, test_path, batch_size=batch_size)
data_train = data.train
data_test = data.test
num_problems = data.num_problems

dkt = DKT(sess, data_train, data_test, num_problems, network_config,
          num_runs=num_runs, num_epochs=num_epochs,
          save_dir_prefix=save_dir_prefix,
          keep_prob=keep_prob, logging=False, save=False)

# load the model
dkt.model.build_graph()
dkt.load_model()

# 调试错误发生点
m1, m2 = dkt.consistency()
msg2 = '& {0} & {1} & {2} & {3:.5f} & {4:.5f} \\\\'.format(network_config['lambda_o'],
                                                           network_config['lambda_w1'],
                                                           network_config['lambda_w2'],
                                                           m1,
                                                           m2)

print(msg)
print(msg2)

# 评估模型 调试错误发生点
print(dkt.evaluate())  # return: auc_n, auc_c, total_loss
print(dkt.waviness())
print("original model: ", dkt_original.reconstruction_accurarcy())
print("regularized model: ", dkt.reconstruction_accurarcy())

#######################################################

problem_seqs_test = dkt.data_test.problem_seqs
correct_seqs_test = dkt.data_test.correct_seqs
targets = []
for i in range(len(problem_seqs_test)):
    num_question_answered = len(problem_seqs_test[i])
    question_ids_answered = sorted(set(problem_seqs_test[i]))
    num_distict_question = len(question_ids_answered)

    if 50 >= num_question_answered >= 30 and 10 >= num_distict_question >= 5:
        targets.append(i)

print(targets)

# if False:
#     plt.figure(figsize=(15, 2))
#     plt.ion()
#     for i, sid in enumerate(targets):
#         plt.figure(i+1)
#         plt.figure(figsize=(15, 2))
#         plt.title('student-id{0}'.format(sid))
#         num_problem_answered = len(problem_seqs_test[sid])
#         problems_ids_answered = sorted(set(problem_seqs_test[sid]))
#         num_distict_question = len(question_ids_answered)
#
#         problem_seq = problem_seqs_test[sid][:num_problem_answered]
#         correct_seq = correct_seqs_test[sid][:num_problem_answered]
#         dkt.plot_output_layer(problem_seq=problem_seq, correct_seq=correct_seq)

# selecting one student to visualize
# good example: 17, 35(able to discover skill relation), 62, 81, 97, 103, 126, 128, 98, 207, 247, 292
# wavy example: 1, 30, 32, 40, 170, 297, 467, 525, 544, 578, 579, 598, 638, 715**, 729
# counter-intuitive example: 30(49(0)), 17(45(0)), 32(12(1)), *40(12(0)), 62(12(0)), 98((7(0))), 544(0(1)), 726(34(0))
sid = 1
num_problem_answered = len(problem_seqs_test[sid])
problems_ids_answered = sorted(set(problem_seqs_test[sid]))
num_distict_question = len(question_ids_answered)

problem_seq = problem_seqs_test[sid][:num_problem_answered]
correct_seq = correct_seqs_test[sid][:num_problem_answered]

print(num_problem_answered)
print(problem_seq)
print(correct_seq)

# %% 错误发生点

output_dkt = dkt.get_output_layer([problem_seq], [correct_seq])
output_original = dkt_original.get_output_layer([problem_seq], [correct_seq])

# %%

plt.figure(figsize=(15, 2))
dkt_fig = dkt_original.plot_output_layer(problem_seq=problem_seq, correct_seq=correct_seq)

figure = dkt_fig.get_figure()
figure.savefig('dkt_id1.pdf', bbox_inches='tight')  # , bbox_extra_artist=[lgd])

# %%

plt.figure(figsize=(15, 2))
dkt.plot_output_layer(problem_seq=problem_seq, correct_seq=correct_seq)

figure = dkt_fig.get_figure()
plt.savefig('dktr_id1.pdf', bbox_inches='tight')

# %%

plt.figure(figsize=(15, 30))
dkt.plot_output_layer(problem_seq=problem_seq, correct_seq=correct_seq,
                      target_problem_ids=range(124))

# %%

problem_seq_aug = problem_seq + [21] * 50
correct_seq_aug = correct_seq + [1, 0] * 25

# %%

plt.figure(figsize=(15, 30))
dkt.plot_output_layer(problem_seq=problem_seq_aug, correct_seq=correct_seq_aug,
                      target_problem_ids=range(124))

# %%

plt.figure(figsize=(15, 2))
problem_seq = [55, 45, 55, 55, 55, 45, 98, 98, 98,
               33, 32, 33, 32, 33, 32, 33, 32, 32,
               33, 32, 33, 32, 33, 32, 33, 33, 32,
               33, 32, 32, 33, 32, 33, 33, 32, 32,
               33, 33, 32, 33, 32, 33, 32, 33, 32,
               33, 33, -1, -1, -1, -1, -1, -1]
correct_seq = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
               0, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1]
dkt.plot_output_layer(problem_seq=problem_seq, correct_seq=correct_seq)

# %%

plt.figure(figsize=(15, 30))
dkt.plot_hidden_layer(problem_seq=problem_seq, correct_seq=correct_seq, layer=0)

# %%

plt.figure(figsize=(10, 30))
dkt.plot_hidden_layer(problem_seq=problem_seq, correct_seq=correct_seq, layer=0)
