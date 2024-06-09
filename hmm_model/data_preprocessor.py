
# data_preprocessor.py
import re
import json
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

class DataPreprocessor:
    def __init__(self):
        self.data_pairs = []
        self.tokenizer = None
        self.vocab_size = None
        self.max_seq_len = None

    # 加载古诗json文件
    def load_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    # 提取json文件中paragraphs里的诗句
    def extract_paragraphs(self, data):
        input_target_pairs = []
        for entry in data:
            paragraphs = entry['paragraphs']
            for para in paragraphs:
                if '□' in para: continue
                sentences = re.split(r'[，。！？,.!?]', para)
                sentences = [sentence for sentence in sentences if sentence]
                if len(sentences) >= 2:
                    for i in range(len(sentences) - 1):
                        if len(sentences[i]) <= 10 and len(sentences[i+1]) <= 10:
                            input_target_pairs.append((sentences[i], sentences[i+1]))

        return input_target_pairs

    # 将诗句分为输入诗句和目标诗句
    def preprocess_data(self, mod = 0):

        if mod == 0:
            # 选取0-9000的唐诗
            print("Train")
            for i in range(0, 50000, 1000):
                self.data_pairs += self.extract_paragraphs(self.load_json(f'../全唐诗/poet.tang.{i}.json'))
            # for i in range(0, 200000, 1000):
            #     self.data_pairs += self.extract_paragraphs(self.load_json(f'../全唐诗/poet.song.{i}.json'))
        else:
            print("Test")
            for i in range(50000, 52000, 1000):
                self.data_pairs += self.extract_paragraphs(self.load_json(f'../全唐诗/poet.tang.{i}.json'))
            # for i in range(200000, 255000, 1000):
            #     self.data_pairs += self.extract_paragraphs(self.load_json(f'../全唐诗/poet.song.{i}.json'))

        # 将数据对分为输入诗句和目标诗句
        inputs, targets = zip(*self.data_pairs)
        # print(inputs)
        # print(targets)
        if mod == 0:
            # 中文按字分词
            self.tokenizer = Tokenizer(char_level=True)
            # 构建词汇表
            self.tokenizer.fit_on_texts(inputs + targets)
        else:
            # 确保这里已经调用 self.load 加载 tokenizer 完成
            pass

        # 上句转为序列
        input_sequences = self.tokenizer.texts_to_sequences(inputs)
        # 下句转为序列
        target_sequences = self.tokenizer.texts_to_sequences(targets)
        # print([len(input_sequence) for input_sequence in input_sequences])
        # print([len(target_sequence) for target_sequence in target_sequences])
        # 最大序列长度
        self.max_seq_len = max(max(len(seq) for seq in input_sequences), 
                               max(len(seq) for seq in target_sequences))
        # 填充序列
        self.input_sequences = pad_sequences(input_sequences, maxlen=self.max_seq_len, padding='post')
        self.target_sequences = pad_sequences(target_sequences, maxlen=self.max_seq_len, padding='post')
        # print(input_sequences)
        # print(target_sequences)
        #词汇表大小
        self.vocab_size = len(self.tokenizer.word_index) + 1
        # print(self.tokenizer.word_index)
        return self.input_sequences, self.target_sequences

    # 保存tokenizer和max_seq_len
    def save(self, tokenizer_path, max_seq_len_path, input_sequences_path, target_sequences_path):
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.word_index, f)
        with open(max_seq_len_path, 'w') as f:
            f.write(str(self.max_seq_len))
        # np.savetxt(input_sequences_path, self.input_sequences)
        # np.savetxt(target_sequences_path, self.target_sequences)
        print("Token saved")

    # 加载tokenizer和max_seq_len
    def load(self, tokenizer_path, max_seq_len_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            word_index = json.load(f)
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.word_index = word_index
        self.tokenizer.index_word = {v: k for k, v in word_index.items()}
        with open(max_seq_len_path, 'r') as f:
            self.max_seq_len = int(f.read())
        self.vocab_size = len(self.tokenizer.word_index) + 1

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    input_sequences, target_sequences = data_preprocessor.preprocess_data()
    data_preprocessor.save('token/tokenizer.json', 'token/max_seq_len.txt', "", "")
