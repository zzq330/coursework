
import sys
from hmm import HMM
if __name__ == "__main__":
    
    n = 8800
    # previous_line = sys.argv[1]
    hmm = HMM(num_iter=0, phase=0, tokenizer_path=f'token/tokenizer_{n}.json', load_path="arg")
    next_line = hmm.generate_next_line("沉舟側畔千帆過")
    print(next_line)
    next_line = hmm.generate_next_line("千杯不醉萬古愁")
    print(next_line)
    next_line = hmm.generate_next_line("君不見黃河之水天上來")
    print(next_line)
    next_line = hmm.generate_next_line("舉頭望明月")
    print(next_line)
    next_line = hmm.generate_next_line("噫噓唏")
    print(next_line)
