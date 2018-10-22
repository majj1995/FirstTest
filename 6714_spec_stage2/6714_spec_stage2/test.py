# COMP6714 Project
# DO NOT MODIFY THIS FILE!!!
from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
from config import config
from model import sequence_labeling
from tqdm import tqdm
from todo import evaluate
import torch
from randomness import apply_random_seed

if __name__ == "__main__":
    _config = config()
    apply_random_seed()

    tag_dict = read_tag_vocab(_config.output_tag_file)
    reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
    word_embedding, word_dict = gen_embedding_from_file(_config.word_embedding_file, _config.word_embedding_dim)
    char_embedding, char_dict = gen_embedding_from_file(_config.char_embedding_file, _config.char_embedding_dim)

    _config.nwords = len(word_dict)
    _config.ntags = len(tag_dict)
    _config.nchars = len(char_dict)

    # read training and development data
    test = DataReader(_config, _config.test_file, word_dict, char_dict, tag_dict, _config.batch_size)

    model = sequence_labeling(_config, word_embedding, char_embedding)
    model.load_state_dict(torch.load(_config.model_file))
    model.eval()
    pred_dev_ins, golden_dev_ins = [], []
    for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in test:
        pred_batch_tag = model.decode(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists, batch_char_mask)
        pred_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(pred_batch_tag.data.tolist(), batch_sentence_len_list.data.tolist())]
        golden_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(batch_tag_index_list.data.tolist(), batch_sentence_len_list.data.tolist())]
        
    f1 = evaluate(golden_dev_ins, pred_dev_ins)
    print(f1)