"""
Data loading and preprocessing for the makemore project.
"""

import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

def create_datasets(input_file, data_config=None, use_lowercase=True):
    """
    Create training and test datasets from input file(s).
    For Nepali names, input_file can be a list of files or a single file.
    """

    if data_config is None:
        from config import DataConfig
        data_config = DataConfig()

    # Handle multiple input files (for male.txt and female.txt)
    if isinstance(input_file, list):
        words = []
        for file in input_file:
            with open(file, 'r', encoding=data_config.encoding) as f:
                data = f.read()
            file_words = data.splitlines()
            file_words = [w.strip() for w in file_words] # get rid of any leading or trailing white space
            file_words = [w for w in file_words if w] # get rid of any empty strings
            words.extend(file_words)
    else:
        # preprocessing of the input text file
        with open(input_file, 'r', encoding=data_config.encoding) as f:
            data = f.read()
        words = data.splitlines()
        words = [w.strip() for w in words] # get rid of any leading or trailing white space
        words = [w for w in words if w] # get rid of any empty strings

    # Always convert to lowercase for optimal model performance with clean a-z training
    words = [w.lower() for w in words]
    
    # Filter to ensure only a-z characters (safety check)
    import re
    words = [w for w in words if re.match(r'^[a-z]+$', w) and len(w) >= 2]
    
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(data_config.max_test_size, int(len(words) * data_config.test_split_ratio))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset
