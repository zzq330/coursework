# predict.py

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from data_preprocessor import DataPreprocessor

# 新建数据预处理对象
data_preprocessor = DataPreprocessor()
# 加载 tokenizer 和 max_seq_len
data_preprocessor.load('tokenizer.json', 'max_seq_len.txt')

# 加载模型
model = load_model('model_saved/poetry_model_0_9000.h5')

# 输入示例句子
input_text = "長安甲第高入雲"  

# 预处理输入
input_sequence = data_preprocessor.tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=data_preprocessor.max_seq_len, padding='post')

# 预测
prediction = model.predict(input_sequence)
predicted_sequence = np.argmax(prediction, axis=-1)[0]

# 将词索引转换回文本
predicted_text = ''.join([data_preprocessor.tokenizer.index_word[idx] for idx in predicted_sequence if idx != 0])
print("Predicted text:", predicted_text)