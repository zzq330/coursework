
import os
import json
import numpy as np
from typing import List

class HMM(object):
    '''
        HMM model implementation
        Use Baum-Welch algorithm (one type of EM algorithm) to train the model
    '''
    def __init__(self, num_iter : int = 20, tokenizer_path : str = r"token/tokenizer_8800.json", load_path = r"hmm_model/arg", phase : int = 0):

        # character encoding dictionary
        with open(tokenizer_path, 'r') as f:
            self.word_to_id = json.load(f)
            self.word_to_id[""] = 0
            self.id_to_word = {value: key for key, value in self.word_to_id.items()}

        # number of hidden states - self defined layer
        self.N = len(self.word_to_id)
        # number of observations - number of characters seen
        self.M = len(self.word_to_id)
        print(self.N, self.M)

        # init hidden state prob vector
        # use init_prob[i]
        self.init_prob = np.zeros(self.N, dtype=float)
        
        # hidden state transition prob vector
        # use trans_prob[i][j]
        self.trans_prob = np.zeros((self.N, self.N), dtype=float)
        
        # emission prob vector
        # use emit_prob[i][j]
        self.emit_prob = np.zeros((self.N, self.M), dtype=float)

        # train iteration num
        self.num_iter = num_iter
        
        # small padding number to avoid mulitple by 0
        self.PAD = 1
        self.threshold = 1000
        self.reduction_factor = 0.5

        # trainin phase or generating phase
        # 0 means generating phase
        # 1 means training phase
        self.phase = phase
        if self.phase == 0:
            self.load_model(self.N, num_iter, load_path=load_path)

    def train(self, observation_sequences : List[List[int]], state_sequences : List[List[int]]):
        '''
            HMM model train starts here
        ''' 
        if self.phase != 1:
            print("Please initialize HMM model using the training phase.")
            return
        # Initialize probabilities based on training data
        self.initialize_probabilities(observation_sequences, state_sequences)
        # Train to modify a little bit
        self.baum_welch(observation_sequences)

    def initialize_probabilities(self, observation_sequences : List[List[int]], state_sequences : List[List[int]]):
        '''
            Initialize all the three arg matrices
        '''
        # Initialize start_prob, trans_prob, emit_prob based on sequences
        # 正向统计的权重为 3
        for obs_seq, state_seq in zip(observation_sequences, state_sequences):

            # Remove all the trailing zeros
            num_zeros_obs = np.sum(obs_seq == 0)
            num_zeros_state = np.sum(state_seq == 0)
            
            if num_zeros_obs <= num_zeros_state:
                obs_seq = obs_seq[obs_seq != 0]
                state_seq = state_seq[:len(obs_seq)]
            else:
                state_seq = state_seq[state_seq != 0]
                obs_seq = obs_seq[:len(state_seq)]
                # print(obs_seq)
                # print(state_seq)
            
            self.init_prob[state_seq[0]] += 3
            for t in range(len(state_seq) - 1):
                self.trans_prob[state_seq[t], state_seq[t + 1]] += 3
                self.emit_prob[state_seq[t], obs_seq[t]] += 3
            self.emit_prob[state_seq[-1], obs_seq[-1]] += 3

        # 反过来在统计一次，虽然没有道理，但算是一种数据增强。。。
        for obs_seq, state_seq in zip(state_sequences, observation_sequences):

            # Remove all the trailing zeros
            num_zeros_obs = np.sum(obs_seq == 0)
            num_zeros_state = np.sum(state_seq == 0)
            
            if num_zeros_obs <= num_zeros_state:
                obs_seq = obs_seq[obs_seq != 0]
                state_seq = state_seq[:len(obs_seq)]
            else:
                state_seq = state_seq[state_seq != 0]
                obs_seq = obs_seq[:len(state_seq)]
                # print(obs_seq)
                # print(state_seq)
            
            self.init_prob[state_seq[0]] += 1
            for t in range(len(state_seq) - 1):
                self.trans_prob[state_seq[t], state_seq[t + 1]] += 1
                self.emit_prob[state_seq[t], obs_seq[t]] += 1
            self.emit_prob[state_seq[-1], obs_seq[-1]] += 1

        # To avoid 0 entries, we add padding into the args
        self.init_prob[self.init_prob == 0.] = self.PAD
        self.trans_prob[self.trans_prob == 0.] = self.PAD
        self.emit_prob[self.emit_prob == 0.] = self.PAD

        # Deal with corner cases
        # np.fill_diagonal(self.trans_prob, self.PAD)
        self.trans_prob[0, :] = self.PAD
        self.trans_prob[:, 0] = self.PAD
        self.reduce_large_values()

        # Normalize to get probabilities
        self.init_prob /= self.init_prob.sum()
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)
        self.emit_prob /= self.emit_prob.sum(axis=1, keepdims=True)
    
    def reduce_large_values(self):
        for i in range(self.trans_prob.shape[0]):
            for j in range(self.trans_prob.shape[1]):
                self.trans_prob[i, j] = np.log2(self.trans_prob[i, j]) + 1

                if self.trans_prob[i, j] > self.threshold:
                    print(self.trans_prob[i, j])
                    self.trans_prob[i, j] *= self.reduction_factor

    def forward(self, observations : List[int]):
        '''
            Forwarding algorithm
            Compute all alpha_t[i], where t is time, i is the state index
        '''
        T = len(observations)
        alpha = np.zeros((T, self.N), dtype=float)
        alpha[0] = self.init_prob * self.emit_prob[:, observations[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = alpha[t-1].dot(self.trans_prob[:, j]) * self.emit_prob[j, observations[t]]
        
        return alpha
    
    def backward(self, observations : List[int]):
        '''
            Backwarding algorithm
            Compute all beta_t[i], where t is time, i is state index
        '''
        T = len(observations)
        beta = np.zeros((T, self.N), dtype=float)
        beta[-1] = np.ones(self.N, dtype=float)
        
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = (beta[t+1] * self.emit_prob[:, observations[t+1]]).dot(self.trans_prob[i])
        
        return beta

    def baum_welch(self, encoded_inputs: List[List[int]]):
        '''
            Baue-Welch algorithm
        '''
        print(len(encoded_inputs))
        for _ in range(self.num_iter):
            gamma_sum = np.zeros(self.N, dtype=float)
            xi_sum = np.zeros((self.N, self.N), dtype=float)
            emission_sum = np.zeros((self.N, self.M), dtype=float)
            
            for observation in encoded_inputs:
                print("Next")

                # Remove all the trailing zeros
                observation = observation[observation != 0]
                T = len(observation)
                
                alpha = self.forward(observation)
                beta = self.backward(observation)
                
                xi = np.zeros((T-1, self.N, self.N), dtype=float)
                for t in range(T-1):
                    denominator = (alpha[t].dot(self.trans_prob) * self.emit_prob[:, observation[t+1]]).dot(beta[t+1])
                    for i in range(self.N):
                        numerator = alpha[t, i] * self.trans_prob[i] * self.emit_prob[:, observation[t+1]] * beta[t+1]
                        xi[t, i] = numerator / denominator
                
                gamma = (alpha * beta) / (alpha * beta).sum(axis=1, keepdims=True)
                gamma_sum += gamma.sum(axis=0)
                xi_sum += xi.sum(axis=0)
                for t in range(T):
                    emission_sum[:, observation[t]] += gamma[t]
            
            self.init_prob = gamma_sum / len(encoded_inputs)
            self.init_prob /= self.init_prob.sum()
            self.trans_prob = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            self.emit_prob = emission_sum / emission_sum.sum(axis=1, keepdims=True)

        self.save_model()
        return

    def save_model(self, save_path = "arg"):
        np.save(os.path.join(save_path, f"initial_prob_n{self.N}_i{self.num_iter}.npy"), self.init_prob)
        np.save(os.path.join(save_path, f"transition_prob_n{self.N}_i{self.num_iter}.npy"), self.trans_prob)
        np.save(os.path.join(save_path, f"emission_prob_n{self.N}_i{self.num_iter}.npy"), self.emit_prob)
        # np.savetxt(os.path.join(save_path, f"initial_prob_n{self.N}_i{self.num_iter}.txt"), self.init_prob)
        # np.savetxt(os.path.join(save_path, f"transition_prob_n{self.N}_i{self.num_iter}.txt"), self.trans_prob)
        # np.savetxt(os.path.join(save_path, f"emission_prob_n{self.N}_i{self.num_iter}.txt"), self.emit_prob)
        print(self.init_prob)
        print(self.trans_prob)
        print(self.emit_prob)
        print("Model saved!")

    def load_model(self, N, I, load_path = "arg"):
        self.init_prob = np.load(os.path.join(load_path, f"initial_prob_n{N}_i{I}.npy"))
        self.trans_prob = np.load(os.path.join(load_path, f"transition_prob_n{N}_i{I}.npy"))
        self.emit_prob = np.load(os.path.join(load_path, f"emission_prob_n{N}_i{I}.npy"))
        # self.init_prob = np.loadtxt(os.path.join(load_path, f"initial_prob_n{N}_i{I}.txt"))
        # self.trans_prob = np.loadtxt(os.path.join(load_path, f"transition_prob_n{N}_i{I}.txt"))
        # self.emit_prob = np.loadtxt(os.path.join(load_path, f"emission_prob_n{N}_i{I}.txt"))
        # print(self.init_prob)
        # print(self.trans_prob)
        # print(self.emit_prob)
        print("Model loaded!")

    def generate_next_line(self, previous_line : str):
        """
            Viterbi algorithm
            Generate next poetry line based on given observations
        """
        if self.phase != 0:
            print("Please initialize HMM model using the generating phase.")
            return

        # Use logarithm scale to avoid float underflow problem
        trans_prob = np.log(self.trans_prob)
        emit_prob = np.log(self.emit_prob)
        init_prob = np.log(self.init_prob)

        T = len(previous_line)
        N = self.N
        viterbi_max_prob = np.zeros((N, T), dtype=float)
        previous_max_id = np.zeros((N, T), dtype=int)

        # Handle the first character separately
        start_wordid = self.word_to_id.get(previous_line[0], None)
        # Deal with the case that the character is not in the dict
        if start_wordid is None:
            viterbi_max_prob[:, 0] = init_prob + np.log(np.ones(N) / N)
        else:
            viterbi_max_prob[:, 0] = init_prob + emit_prob[:, start_wordid]

        previous_max_id[:, 0] = -1

        # Viterbi algorithm dynamic programming
        for step in range(1, T):
            wordid = self.word_to_id.get(previous_line[step], None)
            for state in range(N):
                # For characters not in the dictionary
                if wordid is None:
                    next_prob = viterbi_max_prob[:, step-1] + trans_prob[:, state] + np.log(np.ones(N) / N)[state]
                else:
                    next_prob = viterbi_max_prob[:, step-1] + trans_prob[:, state] + emit_prob[state, wordid]
                
                viterbi_max_prob[state, step] = np.max(next_prob)
                previous_max_id[state, step] = np.argmax(next_prob)

        best_path_id = np.argmax(viterbi_max_prob[:, T-1])

        # Backtracking to find the best path
        best_path = [best_path_id]
        for back_step in range(T-1, 0, -1):
            best_path_id = previous_max_id[best_path_id, back_step]
            best_path.append(best_path_id)

        next_line = [self.id_to_word[id_] for id_ in reversed(best_path)]
        print(next_line)
        return "".join(next_line)
