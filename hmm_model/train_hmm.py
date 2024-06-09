
from hmm import HMM
from data_preprocessor import DataPreprocessor

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    input_sequences, target_sequences = data_preprocessor.preprocess_data()
    data_preprocessor.save(f'token/tokenizer_{data_preprocessor.vocab_size}.json', f'token/max_seq_len_{data_preprocessor.vocab_size}.txt', 'token/input_sequences.json', 'token/target_sequences.json')
    # print(input_sequences)
    # print(target_sequences)
    
    hmm = HMM(num_iter=0, phase=1, tokenizer_path=f'token/tokenizer_{data_preprocessor.vocab_size}.json')
    hmm.train(input_sequences, target_sequences)
