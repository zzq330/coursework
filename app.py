
import random
import json
from flask import Flask, request, jsonify, send_from_directory, render_template
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from BiLSTM.data_preprocessor import DataPreprocessor
from BiLSTM.LSTM import LSTMCell, BidirectionalLSTM
from hmm_model.hmm import HMM
from entropy.entropy_peom import run_my_model, predict

app = Flask(__name__, static_folder='static', template_folder='templates')

# 加载已经训练好的模型和预处理器
model1 = load_model('BiLSTM/model_saved/poetry_model_final_v1.h5',
                    custom_objects={'LSTMCell': LSTMCell,
                                    'BidirectionalLSTM': BidirectionalLSTM})
hmm_model = HMM(num_iter=0, tokenizer_path=r"hmm_model/token/tokenizer_8800.json", phase=0)

# 新建数据预处理对象
data_preprocessor = DataPreprocessor()
# 加载 tokenizer 和 max_seq_len
data_preprocessor.load('BiLSTM/tokenizer/tokenizer_final.json', 'BiLSTM/max_seq_len/max_seq_len_final.txt')

@app.route('/')
def index():
    # return send_from_directory('', 'index.html')
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_sentence = data['sentence']
    input_origin = input_sentence
    # print(input_origin)
    model_type = data['model']
    
    # 预处理输入
    input_sequence = data_preprocessor.tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=data_preprocessor.max_seq_len, padding='post')
    
    # 模型预测
    prediction = [[[]]]
    predicted_text = ""
    if model_type=="NN":
        prediction = model1.predict(input_sequence)
        predicted_sequence = np.argmax(prediction, axis=-1)[0]
    
         # 将词索引转换回文本
        predicted_text = ''.join([data_preprocessor.tokenizer.index_word[idx] for idx in predicted_sequence if idx != 0])
    
    if model_type=="Markov":
        predicted_text = hmm_model.generate_next_line(input_sentence)
    
    if model_type=="Maximum_Entropy":
        vocab = json.load(open('entropy/tokenizer_entropy.json', 'r', encoding='utf-8'))
        input = []

        for i in input_origin:
            try:
                input.append(vocab[i])
            except:
                input.append(random.randint(1, len(vocab)))
        # print(input)
        predicted_text = predict(input, vocab)

    return jsonify({'result': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
