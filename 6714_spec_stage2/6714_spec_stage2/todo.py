import torch
from config import config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

_config = config()


def evaluate(golden_list, predict_list):
    parm1 = 0
    parm2 = 0
    match_num = 0
    F1 = 0
    for idx_i in range(len(golden_list)):
        element1 = golden_list[idx_i]
        element2 = predict_list[idx_i]
        for idx_j in range(len(element1)):
            if element1[idx_j] == element2[idx_j] and "B" in element1[idx_j]:
                index = idx_j + 1
                flag = 1
                while(index < len(element1)-1 and flag == 1):
                    if element1[index] == element2[index] and element1[index] == "O":
                        break
                    elif element1[index] == element2[index] and "I" in element1[index]:
                        index += 1
                    elif element1[index] != element2[index]:
                        flag = 0
                        break
                    elif element1[index] == element2[index] and "B" in element1[index]:
                        break
                if flag == 1:
                    match_num += 1

        for idx_j in range(len(element1)):
            if "B" in element1[idx_j]:
                parm1 += 1
            if "B" in element2[idx_j]:
                parm2 += 1

    if parm1 + parm2 == 1:
        F1 = 1
    else:
        F1 = 2 * match_num / (parm1 + parm2)
    return F1


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + ((1 - forgetgate) * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    size_of_batch_word_len = batch_word_len_lists.size()
    batch_char_index_list = batch_char_index_matrices.view((size_of_batch_word_len[0]*size_of_batch_word_len[1], -1))
    batch_word_len = batch_word_len_lists.view(-1)
    char_embeddings = model.char_embeds(batch_char_index_list)
    parm_index, sorted_batch_word_len_list = model.sort_input(batch_word_len)
    sorterd_embeddings = char_embeddings[parm_index]
    _, original_index = torch.sort(parm_index, descending=False)

    result_sequence = pack_padded_sequence(sorterd_embeddings, lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
    result_sequence, _ = model.char_lstm(result_sequence)
    result_sequence, _ = pad_packed_sequence(result_sequence, batch_first=True)
    sequence = []
    for i in range(len(sorted_batch_word_len_list)):
        sequence.append(torch.cat([result_sequence[i, sorted_batch_word_len_list[i] - 1, :model.config.char_lstm_output_dim], result_sequence[i, 0, model.config.char_lstm_output_dim:]],-1).unsqueeze(0))
    result_sequence = torch.cat(sequence, 0)
    result_sequence = result_sequence[original_index]
    result_sequence = result_sequence.view((size_of_batch_word_len[0], size_of_batch_word_len[1], -1))
    return result_sequence


