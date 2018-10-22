import torch
from config import config

_config = config()


def evaluate(golden_list, predict_list):
    parm1 = len(golden_list)
    parm2 = len(predict_list)
    list1 = []
    list2 = []
    for item_1 in golden_list:
        element = []
        for index in range(len(item_1)):
            if "TAG" in item_1[index]:
                element.append({item_1[index]: index})
            if "HYP" in item_1[index]:
                element.append({item_1[index]: index})
        list1.append(element)
    for item_2 in predict_list:
        element = []
        for index in range(len(item_2)):
            if "TAG" in item_2[index]:
                element.append({item_2[index]: index})
            if "HYP" in item_2[index]:
                element.append({item_2[index]: index})
        list2.append(element)
    list3 = [x for x in list1 if x in list2]
    match_num = len(list3)
    return 2*match_num/(parm1 + parm2)


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
    size_of_batch_char_index = batch_char_index_matrices.size()
    word_max_length = size_of_batch_char_index[2]
    batch_char_index_list = torch.reshape(batch_char_index_matrices, (-1, word_max_length))

    size_of_batch_word_len = batch_word_len_lists.size()
    batch_word_len_list = torch.reshape(batch_word_len_lists, (-1,))
    char_embeddings = model.char_embeds(batch_char_index_list)
    input_embeddings = model.non_recurrent_dropout(char_embeddings)

    parm_index, sorted_batch_world_len_list = model.sort_input(batch_word_len_list)
    sorterd_embeddings = input_embeddings[parm_index]
    _, original_index = model.sort_input(parm_index, descending=False)

    result_sequence = pack_padded_sequence(sorterd_embeddings, lengths=sorted_batch_world_len_list.data.tolist(),
                                           batch_first=True)
    result_sequence, state = model.char_lstm(result_sequence)
    result_sequence, _ = pad_packed_sequence(result_sequence, batch_first=True)
    result_sequence = torch.cat([result_sequence[:, -1, :model.config.char_lstm_output_dim],
                                 result_sequence[:, 0, model.config.char_lstm_output_dim:]], -1)
    result_sequence = result_sequence[original_index]
    result_sequence = model.non_recurrent_dropout(result_sequence)
    result_sequence = torch.reshape(result_sequence, (size_of_batch_word_len[0], size_of_batch_word_len[1], -1))
    return result_sequence

