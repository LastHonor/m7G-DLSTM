import tensorflow as tf
import numpy as np
import os
import time


def mk_dir(path):
    path = path
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


GENE_DICT = {"A": 0, "C": 1, "G": 2, "U": 3}

E_MATRIX = np.identity(4)


def one_hot(gene):
    def tran_digital(ch):
        if ch == 'A':
            return 0
        elif ch == 'C':
            return 1
        elif ch == 'G':
            return 2
        elif ch == "U":
            return 3
        else:
            raise ValueError

    n = len(gene)
    gene_array = np.zeros((n, 4))
    for i in range(n):
        gene_array[i] = E_MATRIX[tran_digital(gene[i])]
    return gene_array


def npf(gene):
    def tran_digital(ch):
        if ch == 'A':
            return np.array([1, 1, 1])
        elif ch == 'C':
            return np.array([0, 0, 1])
        elif ch == 'G':
            return np.array([1, 0, 0])
        elif ch == "U":
            return np.array([0, 1, 0])
        else:
            raise ValueError

    n = len(gene)
    gene_array = np.zeros((n, 3))
    f_array = np.zeros((n, 1))
    a_num = 0
    c_num = 0
    g_num = 0
    u_num = 0
    for i in range(n):
        g = gene[i]
        if g == "A":
            a_num += 1
            num = a_num
        elif g == "C":
            c_num += 1
            num = c_num
        elif g == "G":
            g_num += 1
            num = g_num
        elif g == "U":
            u_num += 1
            num = u_num
        else:
            raise ValueError
        gene_array[i] = tran_digital(g)
        f_array[i] = num / (i + 1)
    gene_array = np.concatenate((gene_array, f_array), axis=1)
    return gene_array


# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)


class LSTM:
    def __init__(self):
        self.va_bz = 1
        self.c_size = [7, 32]
        self.kernel_lstm = [[self.c_size[i - 1] + self.c_size[i], self.c_size[i]] for i in range(1, len(self.c_size))]
        self.kernel_fcn = [[self.c_size[-1] * 2, self.c_size[-1]], [self.c_size[-1], 2]]
        self.e_array = np.identity(2)

        self.level_lstm = len(self.kernel_lstm)
        self.level_fcn = len(self.kernel_fcn)

        self.one_hot_t_list_l = None
        self.one_hot_tb_list_l = None
        self.one_hot_u_list_l = None
        self.one_hot_ub_list_l = None
        self.one_hot_v_list_l = None
        self.one_hot_vb_list_l = None
        self.one_hot_w_list_l = None
        self.one_hot_wb_list_l = None

        self.one_hot_t_list_r = None
        self.one_hot_tb_list_r = None
        self.one_hot_u_list_r = None
        self.one_hot_ub_list_r = None
        self.one_hot_v_list_r = None
        self.one_hot_vb_list_r = None
        self.one_hot_w_list_r = None
        self.one_hot_wb_list_r = None

        self.f_list = None
        self.fb_list = None

    def initial_weight(self, path):
        one_hot_path = path + "/one_hot"
        one_hot_path_l = one_hot_path + "/left"
        one_hot_path_r = one_hot_path + "/right"
        fcn_path = path + "/fcn"

        self.one_hot_t_list_l = [tf.Variable(np.load(one_hot_path_l + "/t%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_tb_list_l = [tf.Variable(np.load(one_hot_path_l + "/tb%d.npy" % i)) for i in
                                  range(self.level_lstm)]
        self.one_hot_u_list_l = [tf.Variable(np.load(one_hot_path_l + "/u%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_ub_list_l = [tf.Variable(np.load(one_hot_path_l + "/ub%d.npy" % i)) for i in
                                  range(self.level_lstm)]
        self.one_hot_v_list_l = [tf.Variable(np.load(one_hot_path_l + "/v%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_vb_list_l = [tf.Variable(np.load(one_hot_path_l + "/vb%d.npy" % i)) for i in
                                  range(self.level_lstm)]
        self.one_hot_w_list_l = [tf.Variable(np.load(one_hot_path_l + "/w%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_wb_list_l = [tf.Variable(np.load(one_hot_path_l + "/wb%d.npy" % i)) for i in
                                  range(self.level_lstm)]

        self.one_hot_t_list_r = [tf.Variable(np.load(one_hot_path_r + "/t%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_tb_list_r = [tf.Variable(np.load(one_hot_path_r + "/tb%d.npy" % i)) for i in
                                  range(self.level_lstm)]
        self.one_hot_u_list_r = [tf.Variable(np.load(one_hot_path_r + "/u%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_ub_list_r = [tf.Variable(np.load(one_hot_path_r + "/ub%d.npy" % i)) for i in
                                  range(self.level_lstm)]
        self.one_hot_v_list_r = [tf.Variable(np.load(one_hot_path_r + "/v%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_vb_list_r = [tf.Variable(np.load(one_hot_path_r + "/vb%d.npy" % i)) for i in
                                  range(self.level_lstm)]
        self.one_hot_w_list_r = [tf.Variable(np.load(one_hot_path_r + "/w%d.npy" % i)) for i in
                                 range(self.level_lstm)]
        self.one_hot_wb_list_r = [tf.Variable(np.load(one_hot_path_r + "/wb%d.npy" % i)) for i in
                                  range(self.level_lstm)]

        self.f_list = [tf.Variable(np.load(fcn_path + "/f%d.npy" % i)) for i in range(self.level_fcn)]
        self.fb_list = [tf.Variable(np.load(fcn_path + "/fb%d.npy" % i)) for i in range(self.level_fcn)]

    def __lstm__(self, x_input, t_list, tb_list, u_list, ub_list, v_list, vb_list, w_list, wb_list):
        n, m, channel = x_input.shape
        h_list = [tf.zeros([n, k[-1]]) for k in self.kernel_lstm]
        c_list = [tf.zeros([n, k[-1]]) for k in self.kernel_lstm]
        x = x_input[:, 0, :]
        for sequence_id in range(m):
            x = x_input[:, sequence_id, :]
            for level in range(self.level_lstm):
                h = h_list[level]
                c = c_list[level]
                t = t_list[level]
                u = u_list[level]
                v = v_list[level]
                w = w_list[level]
                tb = tb_list[level]
                ub = ub_list[level]
                vb = vb_list[level]
                wb = wb_list[level]

                hx = tf.concat([h, x], axis=1)

                hxt = tf.nn.sigmoid(tf.matmul(hx, t) + tb)
                hxu = tf.nn.sigmoid(tf.matmul(hx, u) + ub)
                hxv = tf.nn.tanh(tf.matmul(hx, v) + vb)
                hxw = tf.nn.sigmoid(tf.matmul(hx, w) + wb)

                hxu_hxv = hxu * hxv
                c = c * hxt + hxu_hxv

                c_list[level] = c
                h = tf.nn.tanh(c) * hxw
                h_list[level] = h
                x = h
        return x

    def __net_work__(self, x_l, x_r):
        n = len(x_l)
        m_l = len(x_l[0])
        m_r = len(x_r[0])
        x_one_hot_l = np.zeros((n, m_l, 4))
        x_one_hot_r = np.zeros((n, m_r, 4))
        x_npf_l = np.zeros((n, m_l, 4))
        x_npf_r = np.zeros((n, m_r, 4))

        for i in range(n):
            gene_l = x_l[i]
            gene_r = x_r[i]
            x_one_hot_l[i] = one_hot(gene_l)
            x_one_hot_r[i] = one_hot(gene_r)
            x_npf_l[i] = npf(gene_l)
            x_npf_r[i] = npf(gene_r)

        x_one_hot_l = tf.constant(x_one_hot_l, dtype=tf.float32)
        x_one_hot_r = tf.constant(x_one_hot_r, dtype=tf.float32)
        x_npf_l = tf.constant(x_npf_l[:, :, :-1], dtype=tf.float32)
        x_npf_r = tf.constant(x_npf_r[:, :, :-1], dtype=tf.float32)
        x_one_hot_l = tf.concat([x_one_hot_l, x_npf_l], axis=2)
        x_one_hot_r = tf.concat([x_one_hot_r, x_npf_r], axis=2)

        o_one_hot_l = self.__lstm__(x_one_hot_l,
                                    self.one_hot_t_list_l,
                                    self.one_hot_tb_list_l,
                                    self.one_hot_u_list_l,
                                    self.one_hot_ub_list_l,
                                    self.one_hot_v_list_l,
                                    self.one_hot_vb_list_l,
                                    self.one_hot_w_list_l,
                                    self.one_hot_wb_list_l)
        o_one_hot_r = self.__lstm__(x_one_hot_r,
                                    self.one_hot_t_list_r,
                                    self.one_hot_tb_list_r,
                                    self.one_hot_u_list_r,
                                    self.one_hot_ub_list_r,
                                    self.one_hot_v_list_r,
                                    self.one_hot_vb_list_r,
                                    self.one_hot_w_list_r,
                                    self.one_hot_wb_list_r)

        x_all = tf.concat([o_one_hot_l, o_one_hot_r], axis=1)

        for i in range(self.level_fcn):
            f = self.f_list[i]
            fb = self.fb_list[i]
            x_all = tf.matmul(x_all, f) + fb

            if i == self.level_fcn - 1:
                x_all = tf.nn.softmax(x_all)
            else:
                x_all = tf.nn.softsign(x_all)

        return x_all

    def predict(self, x, g_index_list):
        out_put_list = []
        n = len(x)
        s_time = time.time()
        for i in range(0, n):
            gene_str = x[i]
            g_index = g_index_list[i]

            gene_str_l = [gene_str[:g_index]]
            gene_str_r = [gene_str[g_index + 1:][::-1]]

            output = self.__net_work__(gene_str_l, gene_str_r)
            out_put_list.append(output)

            e_time = time.time()
            during_time = e_time - s_time
            average_time = during_time / (i + 1)
            remaining_time = average_time * (n - (i + 1))
            print("\r predicting: %.02f%%, remaining: %d seconds" % ((i + 1) / n * 100, remaining_time), end="")

        output = tf.concat(out_put_list, axis=0).numpy()
        pre_label = np.argmax(output, axis=1)

        print()

        return output[:, 1], pre_label


def excellent():
    with open("sample.txt", "r") as f:
        sample_string = f.read().split("\n")
    if len(sample_string[-1]) == 0:
        sample_string = sample_string[:-1]
    if len(sample_string[0]) == 0:
        sample_string = sample_string[1:]

    g_index_list = []
    for i, gene_str in enumerate(sample_string):
        assert len(gene_str) >= 3, "The number of nucleobases in %d-th gene sequence is less than 3" % (i + 1)
        if len(gene_str) % 2 == 0:
            g_index_list.append(str(len(gene_str) // 2))
        else:
            g_index_list.append(str(len(gene_str) // 2))

    with open("g_index.txt", "w") as f:
        f.write("\n".join(g_index_list))

    del g_index_list

    if os.path.exists("g_index.txt"):
        with open("g_index.txt", "r") as f:
            g_index_list = f.read().split("\n")
            if len(g_index_list[-1]) == 0:
                g_index_list = g_index_list[:-1]
            if len(g_index_list[0]) == 0:
                g_index_list = g_index_list[1:]
            g_index_list = [int(i) for i in g_index_list]
        assert len(g_index_list) == len(sample_string), \
            "The number of samples in g_index.txt and sample.txt should be equal"

        for i in range(len(g_index_list)):
            gene_str = sample_string[i]
            g = g_index_list[i]
            assert 0 < g < len(gene_str) - 1, \
                "The index of G-site in the %d-th gene sequence is unavailable" % (i + 1)
    else:
        pass

    sample_string = np.array(sample_string)

    lstm = LSTM()
    lstm.initial_weight(path="model")

    sample_pro, sample_label = lstm.predict(sample_string, g_index_list=g_index_list)

    with open("probability_of_m7G.txt", "w") as f:
        sample_pro = [str(i) for i in sample_pro]
        f.write("\n".join(sample_pro))
    with open("result.txt", "w") as f:
        sample_label = ["m7G" if i == 1 else "non-m7G" for i in sample_label]
        f.write("\n".join(sample_label))


if __name__ == '__main__':
    excellent()
