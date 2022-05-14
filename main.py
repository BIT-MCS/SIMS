import os

# importing the libraries
import numpy as np
import random as rn
from conf import *

seed = 1
import torch

rn.seed(seed)
os.environ['PYTHONHASHSEED'] = '3'
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Running the below code every time
import data_op
from dnc_seq2seq.model import *
from utils import LogUtil

log_path = './log'
test_path = ''
log = LogUtil(root_path=log_path, method_name='dnc_fri_new_pytorch')

# hyper-parameters
train_time = CONF['train_time']
test_interval = CONF['test_interval']
n_classes = CONF['n_classes']
nn_layers = CONF['nn_layers']
rnn_size = CONF['rnn_size']
gpu_id = CONF['gpu_id']
test_batch = CONF['test_batch']
mid_layer_size = CONF['mid_layer_size']

learning_rate = CONF['learning_rate']
eps = CONF['eps']
batch_size = CONF['batch_size']


# placeholder


# pip install dm-sonnet
# some functions

def changey(y):
    t = [0]
    t = t * n_classes
    # print y
    t[y] = 1
    return t


def test_evaluate(seq_model, data, step=''):
    MAP = 0.0
    total_recal1 = 0.0
    total_recal2 = 0.0
    total_recal3 = 0.0
    total_recal4 = 0.0
    total_recal5 = 0.0
    ac1 = 0
    ac2 = 0
    ac3 = 0
    ac4 = 0
    ac5 = 0

    x_input, y_input, fr_input = data.get_random_test_batch(test_batch)

    self_presentation, _ = seq_model.encoder_forward(x_input)
    fr_temp = friend_batch(fr_input)

    fr_presentation = []
    for friends in fr_temp:
        temp, _ = seq_model.encoder_forward(friends)
        fr_presentation.append(temp.unsqueeze(dim=1))

    fr_presentation = torch.cat(fr_presentation, dim=1)
    fr_presentation = fr_presentation.reshape(fr_presentation.shape[0], -1)
    # [batch,5*128]

    c_out = seq_model.friend_classif_forward(z=self_presentation, f=fr_presentation)
    _, indexlist_top1 = torch.topk(c_out, 1)
    _, indexlist_top2 = torch.topk(c_out, 2)
    _, indexlist_top3 = torch.topk(c_out, 3)
    _, indexlist_top4 = torch.topk(c_out, 4)
    _, indexlist_top5 = torch.topk(c_out, 5)
    _, indexlist_top_all = torch.topk(c_out, 30)

    indexlist_top1 = indexlist_top1.cpu().numpy()
    indexlist_top2 = indexlist_top2.cpu().numpy()
    indexlist_top3 = indexlist_top3.cpu().numpy()
    indexlist_top4 = indexlist_top4.cpu().numpy()
    indexlist_top5 = indexlist_top5.cpu().numpy()
    indexlist_top_all = indexlist_top_all.cpu().numpy()

    for i in range(len(x_input)):
        the_index = np.where(indexlist_top_all[i] == y_input[i])[0][0]
        MAP += 1.0 / (the_index + 1)

        if (y_input[i] in indexlist_top1[i]):
            ac1 = ac1 + 1

        if (y_input[i] in indexlist_top2[i]):
            ac2 = ac2 + 1

        if (y_input[i] in indexlist_top3[i]):
            ac3 = ac3 + 1

        if (y_input[i] in indexlist_top4[i]):
            ac4 = ac4 + 1

        if (y_input[i] in indexlist_top5[i]):
            ac5 = ac5 + 1
    total_sum = test_batch
    total_recal1 += (ac1 * 1.0) / (total_sum)
    total_recal2 += (ac2 * 1.0) / (total_sum)
    total_recal3 += (ac3 * 1.0) / (total_sum)
    total_recal4 += (ac4 * 1.0) / (total_sum)
    total_recal5 += (ac5 * 1.0) / (total_sum)
    MAP = MAP / (total_sum)

    log_str = step + '\n' + \
              "Testing top 1 recall:" + "{:.5f}".format(total_recal1) + '\n' + \
              "Testing top 2 recall:" + "{:.5f}".format(total_recal2) + '\n' + \
              "Testing top 3 recall:" + "{:.5f}".format(total_recal3) + '\n' + \
              "Testing top 4 recall:" + "{:.5f}".format(total_recal4) + '\n' + \
              "Testing top 5 recall:" + "{:.5f}".format(total_recal5) + '\n' + \
              "Testing top 1 F1-score:" + "{:.5f}".format(total_recal1) + '\n' + \
              "Testing top 2 F1-score:" + "{:.5f}".format(2.0 * total_recal2 / 3) + '\n' + \
              "Testing top 3 F1 -score:" + "{:.5f}".format(2.0 * total_recal3 / 4) + '\n' + \
              "Testing top 4 F1 -score:" + "{:.5f}".format(2.0 * total_recal4 / 5) + '\n' + \
              "Testing top 5 F1-score:" + "{:.5f}".format(2.0 * total_recal5 / 6) + '\n' + \
              "Testing MAP:" + "{:.5f}".format(MAP)
    log.record_report(
        log_path
    )
    return MAP, log_str


def evaluate(seq_model, data, step=''):
    MAP = 0.0
    total_recal1 = 0.0
    total_recal2 = 0.0
    total_recal3 = 0.0
    total_recal4 = 0.0
    total_recal5 = 0.0
    ac1 = 0
    ac2 = 0
    ac3 = 0
    ac4 = 0
    ac5 = 0
    res_list = []
    while True:
        x_input, y_input, fr_input = data.get_batch_test_dataset(batch_size=512)
        if x_input is None:
            break
        self_presentation, _ = seq_model.encoder_forward(x_input)
        fr_temp = friend_batch(fr_input)

        fr_presentation = []
        for friends in fr_temp:
            temp, _ = seq_model.encoder_forward(friends)
            fr_presentation.append(temp.unsqueeze(dim=1))

        fr_presentation = torch.cat(fr_presentation, dim=1)
        fr_presentation = fr_presentation.reshape(fr_presentation.shape[0], -1)
        # [batch,5*128]

        c_out = seq_model.friend_classif_forward(z=self_presentation, f=fr_presentation)
        _, indexlist_top1 = torch.topk(c_out, 1)
        _, indexlist_top2 = torch.topk(c_out, 2)
        _, indexlist_top3 = torch.topk(c_out, 3)
        _, indexlist_top4 = torch.topk(c_out, 4)
        _, indexlist_top5 = torch.topk(c_out, 5)
        _, indexlist_top_all = torch.topk(c_out, 30)

        indexlist_top1 = indexlist_top1.cpu().numpy()
        indexlist_top2 = indexlist_top2.cpu().numpy()
        indexlist_top3 = indexlist_top3.cpu().numpy()
        indexlist_top4 = indexlist_top4.cpu().numpy()
        indexlist_top5 = indexlist_top5.cpu().numpy()
        indexlist_top_all = indexlist_top_all.cpu().numpy()

        for i in range(len(x_input)):
            the_index = np.where(indexlist_top_all[i] == y_input[i])[0][0]
            MAP += 1.0 / (the_index + 1)
            one_map = 1.0 / (the_index + 1)
            one_acc1 = 0
            one_acc2 = 0
            one_acc3 = 0
            one_acc4 = 0
            one_acc5 = 0
            if (y_input[i] in indexlist_top1[i]):
                ac1 = ac1 + 1
                one_acc1 = 1

            if (y_input[i] in indexlist_top2[i]):
                ac2 = ac2 + 1
                one_acc2 = 1

            if (y_input[i] in indexlist_top3[i]):
                ac3 = ac3 + 1
                one_acc3 = 1

            if (y_input[i] in indexlist_top4[i]):
                ac4 = ac4 + 1
                one_acc4 = 1

            if (y_input[i] in indexlist_top5[i]):
                ac5 = ac5 + 1
                one_acc5 = 1
            res_list.append((one_map, one_acc1, one_acc2, one_acc3, one_acc4, one_acc5))
    total_sum = len(data.x_input)
    total_recal1 += (ac1 * 1.0) / (total_sum)
    total_recal2 += (ac2 * 1.0) / (total_sum)
    total_recal3 += (ac3 * 1.0) / (total_sum)
    total_recal4 += (ac4 * 1.0) / (total_sum)
    total_recal5 += (ac5 * 1.0) / (total_sum)
    MAP = MAP / (total_sum)
    print("Testing top 1 recall:" + "{:.5f}".format(total_recal1))
    print("Testing top 2 recall:" + "{:.5f}".format(total_recal2))
    print("Testing top 3 recall:" + "{:.5f}".format(total_recal3))
    print("Testing top 4 recall:" + "{:.5f}".format(total_recal4))
    print("Testing top 5 recall:" + "{:.5f}".format(total_recal5))

    print("Testing top 1 F1-score:" + "{:.5f}".format(total_recal1))
    print("Testing top 2 F1-score:" + "{:.5f}".format(2.0 * total_recal2 / 3))
    print("Testing top 3 F1 -score:" + "{:.5f}".format(2.0 * total_recal3 / 4))
    print("Testing top 4 F1 -score:" + "{:.5f}".format(2.0 * total_recal3 / 5))
    print("Testing top 5 F1-score:" + "{:.5f}".format(2.0 * total_recal5 / 6))
    print("Testing MAP:" + "{:.5f}".format(MAP))

    log.record_report(
        step + '\n' +
        "Testing top 1 recall:" + "{:.5f}".format(total_recal1) + '\n' +
        "Testing top 2 recall:" + "{:.5f}".format(total_recal2) + '\n' +
        "Testing top 3 recall:" + "{:.5f}".format(total_recal3) + '\n' +
        "Testing top 4 recall:" + "{:.5f}".format(total_recal4) + '\n' +
        "Testing top 5 recall:" + "{:.5f}".format(total_recal5) + '\n' +

        "Testing top 1 F1-score:" + "{:.5f}".format(total_recal1) + '\n' +
        "Testing top 2 F1-score:" + "{:.5f}".format(2.0 * total_recal2 / 3) + '\n' +
        "Testing top 3 F1 -score:" + "{:.5f}".format(2.0 * total_recal3 / 4) + '\n' +
        "Testing top 4 F1 -score:" + "{:.5f}".format(2.0 * total_recal3 / 5) + '\n' +
        "Testing top 5 F1-score:" + "{:.5f}".format(2.0 * total_recal5 / 6) + '\n' +
        "Testing MAP:" + "{:.5f}".format(MAP)
    )
    return MAP, res_list


def friend_batch(fr_input):
    fr1 = []
    fr2 = []
    fr3 = []
    fr4 = []
    fr0 = []
    for fr in fr_input:
        fr0.append(fr[0])
        fr1.append(fr[1])
        fr2.append(fr[2])
        fr3.append(fr[3])
        fr4.append(fr[4])
    return [fr0, fr1, fr2, fr3, fr4]


def train():
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    MAXMAP = 0

    data = data_op.data_op()

    seq_model = SequentialModel(gpu_id).to('cuda:' + str(gpu_id))
    crit2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=learning_rate,
                                 eps=eps,
                                 weight_decay=1e-6
                                 )
    for step in range(train_time + 1):
        x_input, y_input, fr_input = data.get_batch_train_dataset(batch_size)

        y_input = torch.tensor(y_input, dtype=torch.long, device='cuda:' + str(gpu_id))

        self_presentation, _ = seq_model.encoder_forward(x_input)
        fr_temp = friend_batch(fr_input)

        fr_presentation = []
        for friends in fr_temp:
            temp, _ = seq_model.encoder_forward(friends)
            fr_presentation.append(temp.unsqueeze(dim=1))

        fr_presentation = torch.cat(fr_presentation, dim=1)
        fr_presentation = fr_presentation.reshape(fr_presentation.shape[0], -1)
        # [batch,5*128]

        c_out = seq_model.friend_classif_forward(z=self_presentation, f=fr_presentation)

        optimizer.zero_grad()

        model_loss = crit2(c_out, y_input)
        model_loss.backward()
        optimizer.step()

        print(step, 'model loss:%f' % model_loss, )
        log.record_loss(loss=model_loss)
        if step % test_interval == 0:
            seq_model.eval()
            with torch.no_grad():
                MAP = evaluate(seq_model, data, str(step))
            seq_model.train()
            if MAXMAP < MAP:
                MAXMAP = MAP
                log.save_model(seq_model)




if __name__ == '__main__':
    train()
