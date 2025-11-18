import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        encoder_inputs = load_lines(nl_path)

        encoder_inputs = [f"translate English to SQL: {nl}" for nl in encoder_inputs]
        tokenized_encoder_inputs = tokenizer(encoder_inputs, return_tensors='pt', padding=True, truncation=True)

        decoder_inputs = None
        decoder_targets = None

        if split != 'test':
            sql_path = os.path.join(data_folder, f"{split}.sql")
            decoder_outputs = load_lines(sql_path)
            
            tokenized_decoder_outputs = tokenizer(decoder_outputs, return_tensors='pt', padding=True, truncation=True)
            
            decoder_inputs = torch.cat([torch.full((tokenized_decoder_outputs.input_ids.shape[0], 1), tokenizer.pad_token_id), tokenized_decoder_outputs.input_ids], dim=1)
            decoder_targets = torch.cat([tokenized_decoder_outputs.input_ids, torch.full((tokenized_decoder_outputs.input_ids.shape[0], 1), tokenizer.pad_token_id)], dim=1)

        return tokenized_encoder_inputs, decoder_inputs, decoder_targets
    
    def __len__(self):
        return len(self.encoder_inputs.input_ids)

    def __getitem__(self, idx):
        if self.split != 'test':
            return self.encoder_inputs.input_ids[idx], self.encoder_inputs.attention_mask[idx], self.decoder_inputs[idx], self.decoder_targets[idx]
        else:
            return self.encoder_inputs.input_ids[idx], self.encoder_inputs.attention_mask[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask, decoder_inputs, decoder_targets = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    initial_decoder_inputs = decoder_inputs[:, :1]

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    
    # For T5, the initial decoder input is the pad token.
    initial_decoder_inputs = torch.full((len(encoder_ids), 1), PAD_IDX)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x