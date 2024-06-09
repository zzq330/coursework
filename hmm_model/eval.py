
import nltk
import torch
import numpy as np
from data_preprocessor import DataPreprocessor
from transformers import BertTokenizer, BertModel
from hmm import HMM

# 下载nltk所需数据
nltk.download('punkt')

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1)

def evaluate_poem(reference, candidate):
    # 计算BERT语义相似度
    ref_embedding = get_sentence_embedding(reference, model, tokenizer)
    gen_embedding = get_sentence_embedding(candidate, model, tokenizer)
    similarity = torch.cosine_similarity(ref_embedding, gen_embedding).item()
    return similarity

def main():
    n = 8800
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load(f'token/tokenizer_{n}.json', f'token/max_seq_len_{n}.txt')
    input_sequences, target_sequences = data_preprocessor.preprocess_data(mod=1)
    print(len(input_sequences))

    # 加载 HMM 模型
    hmm = HMM(num_iter=0, phase=0, tokenizer_path=f'token/tokenizer_{n}.json', load_path="arg")
    similarities = []
    
    for input, target in zip(input_sequences, target_sequences):
        if len(input) == 0 or len(target) == 0: continue
        try:
            # Trim the trailing zeros
            input = np.array(input)
            input = input[input != 0].tolist()
            input = [hmm.id_to_word[character] for character in input]
            print(f"input: {input}")
            generated_target = hmm.generate_next_line(input)

            target = [hmm.id_to_word[char_id] for char_id in target]
            target = "".join(target)
            print(target, generated_target)

            similarity = evaluate_poem(target, generated_target)
            print(f"BERT语义相似度: {similarity}")
            similarities.append(similarity)
        except:
            continue

    avg = sum(similarities) / len(similarities)
    print(f"Avg: {avg}")

if __name__ == "__main__":
    main()
