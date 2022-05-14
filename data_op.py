import random
import numpy as np
import os


class ItemsInfo:
    def __init__(self, category, time):
        self.category = category
        self.time = time


class data_op(object):
    def __init__(self):
        print('dataset: ciao')
        self.train_dic, self.test_dic, self.friend_dic, self.train_user, self.test_user = self.get_reviews()
        self.train_starttime = 959670000
        self.train_endtime = 1172732400
        self.x_input = []
        self.y_input = []
        self.fr_input = []
        self.test_counter = 0

    def get_batch_train_dataset(self, batch_size=128):
        x_input = []
        y_input = []
        fr_input = []
        temp_dic = {}
        rand_time = random.randint(self.train_starttime, self.train_endtime)
        for key, val in self.train_dic.items():
            temp = []
            for var in val:
                if var.time <= rand_time:
                    temp.append(var.category)
            temp_dic[key] = temp
        for _ in range(batch_size):
            rand_index = random.randint(0, len(self.train_user) - 1)
            rand_user = self.train_user[rand_index]
            while len(temp_dic[rand_user]) == 0:
                rand_index = random.randint(0, len(self.train_user) - 1)
                rand_user = self.train_user[rand_index]
            x_input.append((temp_dic[rand_user])[0:-1])
            y_input.append(temp_dic[rand_user][-1])
            sum = 0
            i = 0
            temp_friend = []
            if rand_user in self.friend_dic:
                while i < 5 and i < len(self.friend_dic[rand_user]):
                    friendid = self.friend_dic[rand_user][i]
                    if friendid in self.train_dic:
                        sum += 1
                        temp_friend.append(temp_dic[friendid])
                    i += 1
            if sum < 5:
                for _ in range(5 - sum):
                    temp_friend.append([])
            fr_input.append(temp_friend)
        return x_input, y_input, fr_input

    def init_test(self):
        x_input = []
        y_input = []
        fr_input = []
        for index in range(len(self.test_user)):
            # rand_ = random.randint(0,  - 1)
            rand_user = self.test_user[index]
            x_input.append([var.category for var in (self.test_dic[rand_user])[0:-1]])
            y_input.append((self.test_dic[rand_user])[-1].category)
            sum = 0
            i = 0
            temp_friend = []
            if rand_user in self.friend_dic:
                while i < 5 and i < len(self.friend_dic[rand_user]):
                    friendid = self.friend_dic[rand_user][i]
                    if friendid in self.test_dic:
                        sum += 1
                        temp_friend.append([var.category for var in self.test_dic[friendid]])
                    i += 1
            if sum < 5:
                for _ in range(5 - sum):
                    temp_friend.append([])
            fr_input.append(temp_friend)
        self.x_input = x_input
        self.y_input = y_input
        self.fr_input = fr_input
        if not os.path.exists('x_input.npy'):
            np.save('x_input.npy', x_input)
            np.save('y_input.npy', y_input)
            np.save('fr_input.npy', fr_input)
        else:
            self.x_input = np.load('x_input.npy', allow_pickle=True)
            self.y_input = np.load('y_input.npy', allow_pickle=True)
            self.fr_input = np.load('fr_input.npy', allow_pickle=True)

    # def get_batch_test_dataset(self):
    #     if len(self.x_input) == 0:
    #         self.init_test()
    #
    #     return self.x_input, self.y_input, self.fr_input
    def get_batch_test_dataset(self, batch_size=128):
        if len(self.x_input) == 0:
            self.init_test()

        print(self.test_counter, len(self.x_input))
        if self.test_counter >= len(self.x_input):
            self.test_counter = 0
            return (None, None, None)
        st = self.test_counter
        ed = min(st + batch_size, len(self.x_input))
        self.test_counter += batch_size
        x_input = self.x_input[st:ed]
        y_input = self.y_input[st:ed]
        fr_input = self.fr_input[st:ed]

        return x_input, y_input, fr_input

    def get_random_test_batch(self, batch_size):
        if len(self.x_input) == 0:
            self.init_test()
        x_input = []
        y_input = []
        fr_input = []
        for i in range(batch_size):
            idx = np.random.randint(0, len(self.x_input), 1)[0]
            x_input.append(self.x_input[idx])
            y_input.append(self.y_input[idx])
            fr_input.append(self.fr_input[idx])
        return x_input, y_input, fr_input

    def get_reviews(self):
        train_path = 'ciao/ciao_train.txt'
        test_path = 'ciao/ciao_test.txt'
        friend_path = 'ciao/ciao_friendship.txt'
        train_dic = {}
        test_dic = {}
        friend_dic = {}
        train_user = []
        test_user = []
        with open(train_path, 'r') as f:
            for line in f.readlines():
                sub_str = line.split(' ')
                index = int(sub_str[0])
                train_user.append(index)
                train_dic[index] = []
                for var in sub_str[1:]:
                    temp = var.split(',')
                    info = ItemsInfo(int(temp[0]) - 1, int(temp[1]))
                    train_dic[index].append(info)
        with open(test_path, 'r') as f:
            for line in f.readlines():
                sub_str = line.split(' ')
                index = int(sub_str[0])
                test_user.append(index)
                test_dic[index] = []
                for var in sub_str[1:]:
                    temp = var.split(',')
                    info = ItemsInfo(int(temp[0]) - 1, int(temp[1]))
                    test_dic[index].append(info)
        with open(friend_path, 'r') as f:
            for line in f.readlines():
                sub_str = line.split(' ')
                index = int(sub_str[0])
                friend_dic[index] = []
                for var in sub_str[1:]:
                    friend_dic[index].append(int(var))
        return train_dic, test_dic, friend_dic, train_user, test_user
