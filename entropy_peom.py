import numpy as np
import json
import tqdm
np.random.seed(10)
# train.py
from data_preprocessor import DataPreprocessor


class MyMaxEntropy(object):
    def __init__(self, lr=0.0001):
        """
        最大熵模型的实现，为了方便理解，尽可能的将参数都存储为字典形式
        :param lr: 学习率，默认值为0.0001

        其他参数：
        :param w: 模型的参数，字典
        :param N: 样本数量
        :param label: 标签空间
        :param hat_p_x: 边缘分布P(X)的经验分布
        :param hat_p_x_y: 联合分布P(X,Y)的经验分布
        :param E_p: 特征函数f(x,y)关于模型P(X|Y)与经验分布hatP(X)的期望值
        :param E_hat_p: 特征函数f(x,y)关于经验分布hatP(X,Y)的期望值
        :param eps: 一个接近于0的正数极小值，这个值放在log的计算中，防止报错
        """
        self.lr = lr
        self.params = {'w': None}

        self.N = None
        self.label = None

        self.hat_p_x = {}
        self.hat_p_x_y = {}

        self.E_p = {}
        self.E_hat_p = {}

        self.eps = np.finfo(np.float32).eps


    def _init_params(self):
        """
        随机初始化模型参数w
        :return:
        """
        w = {}
        for key in self.hat_p_x_y.keys():
            w[key] = np.random.rand()
        self.params['w'] = w

    def _rebuild_X(self, X):
        """
        为了自变量的差异化处理，重新命名自变量
        :param X: 原始自变量
        :return:
        """
        X_result = []
        for x in X:
            X_result.append([y_s + '_' + x_s for x_s, y_s in zip(x, self.X_columns)])
        return X_result

    def _build_mapping(self, X, Y):
        """
        求取经验分布，参照公式(1)(2)
        :param X: 训练样本的输入值
        :param Y: 训练样本的输出值
        :return:
        """
        for x, y in zip(X, Y):
            for x_s in x:
                if x_s in self.hat_p_x.keys():
                    self.hat_p_x[x_s] += 1
                else:
                    self.hat_p_x[x_s] = 1
                if (x_s, y) in self.hat_p_x_y.keys():
                    self.hat_p_x_y[(x_s, y)] += 1
                else:
                    self.hat_p_x_y[(x_s, y)] = 1

        self.hat_p_x = {key: count / self.N for key, count in self.hat_p_x.items()}
        self.hat_p_x_y = {key: count / self.N for key, count in self.hat_p_x_y.items()}

    def _cal_E_hat_p(self):
        """
        计算特征函数f(x,y)关于经验分布hatP(X,Y)的期望值，参照公式(3)
        :return:
        """
        self.E_hat_p = self.hat_p_x_y


    def _cal_E_p(self, X):
        """
        计算特征函数f(x,y)关于模型P(X|Y)与经验分布hatP(X)的期望值，参照公式(4)
        :param X:
        :return:
        """
        for key in self.params['w'].keys():
            self.E_p[key] = 0
        for x in X:
            p_y_x = self._cal_prob(x)
            for x_s in x:
                for (p_y_x_s, y) in p_y_x:
                    if (x_s, y) not in self.E_p.keys():
                        continue
                    self.E_p[(x_s, y)] += (1/self.N) * p_y_x_s

    def _cal_p_y_x(self, x, y):
        """
        计算模型条件概率值，参照公式(9)的指数部分
        :param x: 单个样本的输入值
        :param y: 单个样本的输出值
        :return:
        """

        sum = 0.0
        for x_s in x:
            sum += self.params['w'].get((x_s, y), 0)
        return np.exp(sum), y


    def _cal_prob(self, x):
        """
        计算模型条件概率值，参照公式(9)
        :param x: 单个样本的输入值
        :return:
        """
        p_y_x = [(self._cal_p_y_x(x, y)) for y in self.label]
        sum_y = np.sum([p_y_x_s for p_y_x_s, y in p_y_x])
        return [(p_y_x_s / sum_y, y) for p_y_x_s, y in p_y_x]

    def save_model(self, model_path):
        with open(model_path, 'w') as f:
            for key, value in self.params.items():
                if key == 'w':
                    for key, value in value.items():
                        f.write(json.dumps({str(key): value}) + '\n')
                else:
                    f.write(json.dumps({key: value}) + '\n')

    def load_model(self, model_path, label, X_columns):
        self.label = label
        self.X_columns = X_columns
        
        w = {}
        with open(model_path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                for key, value in line.items():
                    w[eval(key)] = value
        self.params['w'] = w

    def fit(self, X, X_columns, Y, label, max_iter=20000, save_path=None):
        """
        模型训练入口
        :param X: 训练样本输入值
        :param X_columns: 训练样本的columns
        :param Y: 训练样本的输出值
        :param label: 训练样本的输出空间
        :param max_iter: 最大训练次数
        :return:
        """
        self.N = len(X)
        self.label = label
        self.X_columns = X_columns
        # import pdb; pdb.set_trace()
        X = self._rebuild_X(X)

        self._build_mapping(X, Y)

        self._cal_E_hat_p()

        self._init_params()

        for iter in tqdm.tqdm(range(max_iter)):

            self._cal_E_p(X)

            for key in self.params['w'].keys():
                sigma = self.lr * np.log(self.E_hat_p.get(key, self.eps) / self.E_p.get(key, self.eps))
                self.params['w'][key] += sigma

        if save_path is not None:
            self.save_model(save_path)

    def predict(self, X):
        """
        预测结果
        :param X: 样本
        :return:
        """
        X = self._rebuild_X(X)
        result_list = []

        for x in X:
            max_result = 0
            y_result = self.label[0]
            p_y_x = self._cal_prob(x)
            for (p_y_x_s, y) in p_y_x:
                if p_y_x_s > max_result:
                    max_result = p_y_x_s
                    y_result = y
            result_list.append((max_result, y_result))
        return result_list


def run_my_model(dataset, vocab, data_preprocessor):
    # data_set = [['物', '外', '幽', '奇', '物'],
    #            ['外', '幽', '奇', '物', '吟'],
    #            ['幽', '奇', '物', '吟', '看'],
    #            ['奇', '物', '吟', '看', '復'],
    #            ['物', '吟', '看', '復', '歎'],
    #            ['吟', '看', '復', '歎', '吁'] ]
    data_set = dataset
    columns = [1,2,3,4,5,6]
    labels = list(vocab.values())
    # import pdb; pdb.set_trace()
    data_set = [[str(i) for i in j] for j in data_set]
    columns = [str(i) for i in columns]
    labels = [str(i) for i in labels]

    X = [i[:-1] for i in data_set]
    X_columns = columns[:-1]
    Y = [i[-1] for i in data_set]
    # print(X)
    # print(Y)

    my = MyMaxEntropy()
    train_len = 0
    test_len = 5
    train_X = X[:train_len]
    test_X = X[train_len:train_len+test_len]
    train_Y = Y[:train_len]
    test_Y = Y[train_len:train_len+test_len]
    
    path = "D:\AppData/python/peom_generate/coursework/model_saved/entropy_final.jsonl"
    # my.fit(train_X, X_columns, train_Y, label=labels, max_iter=100, save_path=path)
    my.load_model(path, labels, X_columns)

    # print(my.params)

    pred_Y= my.predict(test_X)
    print('result: ')
    # import pdb; pdb.set_trace()
    input_index = test_X[0]
    truth_index = test_Y
    pred_index = [i[1] for i in pred_Y]
    input_text = ''.join([data_preprocessor.tokenizer.index_word[int(idx)] for idx in input_index if int(idx) != 0])
    truth_text = ''.join([data_preprocessor.tokenizer.index_word[int(idx)] for idx in truth_index if int(idx) != 0] )
    pred_text = ''.join([data_preprocessor.tokenizer.index_word[int(idx)] for idx in pred_index if int(idx) != 0] )
    print(input_text, truth_text, pred_text)
    # for i in range(len(test_Y)):
    #     print(test_Y[i], pred_Y[i])
 

def get_train_dataset(input_sequences, target_sequences):
    dataset = []
    for i in range(len(input_sequences)):
        seq = []
        for word in input_sequences[i]:
            if word != 0:
                seq.append(word)
        for word in target_sequences[i]:
            if word != 0:
                seq.append(word)
        # seq.shape: (10,)
        input_len = 5
        for i in range(len(seq)-input_len):
            dataset.append(seq[i:i+input_len+1])
    return dataset

def predict(X_input, vocab):
    columns = [1,2,3,4,5,6]
    labels = list(vocab.values())

    X_input = [str(i) for i in X_input[0]]
    columns = [str(i) for i in columns]
    labels = [str(i) for i in labels]
    X_columns = columns[:-1]

    my = MyMaxEntropy()
    path = "D:\AppData/python/peom_generate/coursework/model_saved/entropy_final.jsonl"
    my.load_model(path, labels, X_columns)

    pred_Y= my.predict(X_input)
    print('result: ')
    
    input_index = X_input
    pred_index = [i[1] for i in pred_Y]
    input_text = ''.join([data_preprocessor.tokenizer.index_word[int(idx)] for idx in input_index if int(idx) != 0])
    pred_text = ''.join([data_preprocessor.tokenizer.index_word[int(idx)] for idx in pred_index if int(idx) != 0] )
    print(input_text, pred_text)


if __name__ == '__main__':
    # 新建数据预处理对象
    data_preprocessor = DataPreprocessor()
    # 构造输入诗句和目标诗句
    # shape: (52921, 10)
    input_sequences, target_sequences = data_preprocessor.preprocess_data()
    # 保存 tokenizer 和 max_seq_len
    data_preprocessor.save('D:\AppData\python\peom_generate\coursework\\token/tokenizer_entropy.json', 'D:\AppData\python\peom_generate\coursework\\token/max_seq_len_entropy.txt')
    vocab = json.load(open('D:\AppData\python\peom_generate\coursework\\token/tokenizer_entropy.json', 'r', encoding='utf-8'))
    dataset = get_train_dataset(input_sequences, target_sequences)
    # import pdb; pdb.set_trace()
    
    import pdb; pdb.set_trace()
    # run_my_model(dataset=dataset, vocab=vocab, data_preprocessor=data_preprocessor)
    input_text = '海上生明月'
    input_sequence = data_preprocessor.tokenizer.texts_to_sequences([input_text])
    predict(input_sequence, vocab)