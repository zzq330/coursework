# train.py

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from BiLSTM.data_preprocessor import DataPreprocessor
from BiLSTM.LSTM import BidirectionalLSTM

# 新建数据预处理对象
data_preprocessor = DataPreprocessor()
# 构造输入诗句和目标诗句
input_sequences, target_sequences = data_preprocessor.preprocess_data()
# 保存 tokenizer 和 max_seq_len
data_preprocessor.save('tokenizer/tokenizer_final.json', 'max_seq_len/max_seq_len_final.txt')

# 模型定义和训练
model = Sequential()
model.add(Embedding(input_dim=data_preprocessor.vocab_size, output_dim=128, input_length=data_preprocessor.max_seq_len))
model.add(BidirectionalLSTM(units=256))
# model.add(BidirectionalLSTM(units=256))
# model.add(Bidirectional(LSTM(256, return_sequences=True)))
# model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(data_preprocessor.vocab_size, activation='softmax')))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(input_sequences, target_sequences, epochs=200, batch_size=512, validation_split=0.2)

# 保存模型到文件
model.save('model_saved/poetry_model_final.h5')