import torch
import torch.nn as nn
from dnc_seq2seq.dnc import DNC
from utils import *


class SequentialModel(torch.nn.Module):
    def __init__(self, gpu_id, hidden_size=128, n_class=30, embedding_size=15, num_fri=5):
        super(SequentialModel, self).__init__()
        # self.rnn = SDNC(
        #     input_size=64,
        #     hidden_size=128,
        #     rnn_type='lstm',
        #     num_layers=4,
        #     nr_cells=100,
        #     cell_size=32,
        #     read_heads=4,
        #     sparse_reads=4,
        #     batch_first=True,
        #     gpu_id=0
        # )
        self.hidden_size = hidden_size
        self.rnn = DNC(
            input_size=embedding_size,
            hidden_size=hidden_size,
            rnn_type='lstm',
            num_layers=16,
            nr_cells=8,
            cell_size=8,
            # num_layers=16,
            # nr_cells=8,
            # cell_size=8,
            read_heads=1,
            # sparse_reads=1,
            batch_first=True,
            gpu_id=gpu_id,
            clip=20

        )
        self.n_class = n_class + 3
        init_ = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('linear'))
        self.PAD = n_class
        self.GO = n_class + 1
        self.EOF = n_class + 2

        self.embdedding_size = embedding_size
        self.embeds = nn.Embedding(self.n_class, embedding_size)

        self.gpu_id = gpu_id

        self.friends_linear = nn.Sequential(
            init_(nn.Linear(self.hidden_size * num_fri, self.hidden_size)),
            nn.ReLU(),
        )

        self.clasif_linear = nn.Sequential(
            init_(nn.Linear(self.hidden_size * 2, self.hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_size, n_class)),
            # nn.Softmax(dim=1),
        )
        # (controller_hidden, memory, read_vectors) = (None, None, None)
        #
        # output, (controller_hidden, memory, read_vectors) = \
        #     rnn(torch.randn(10, 4, 64), (controller_hidden, memory, read_vectors), reset_experience=True)

    def encoder_forward(self, x_input):

        x_source_sequence_length, max_length = self.get_sequence_length(x_input)
        if max_length == 0:
            x_input = [[0 for _ in range(self.hidden_size)] for _ in range(len(x_input))]

        x_input = self.pad_sentence_batch(x_input, self.PAD)

        x_input = self.word_to_embedding(x_input)

        (controller_hidden, memory, read_vectors) = (None, None, None)

        output, (controller_hidden, memory, read_vectors) = \
            self.rnn(x_input, (controller_hidden, memory, read_vectors), reset_experience=True)
        return output[:, -1, :], (controller_hidden, memory, read_vectors)

    def friend_classif_forward(self, z, f):
        f_out = self.friends_linear(f)
        c_out = self.clasif_linear(torch.cat([z, f_out], dim=1))
        return c_out

    def get_sequence_length(self, x_input):
        sequence_length = []
        for each in x_input:
            sequence_length.append(len(each))
        max_length = max(sequence_length)
        return sequence_length, max_length

    def pad_sentence_batch(self, sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def word_to_embedding(self, x_input):
        embeddings = []
        for sent in x_input:
            sent_tensor = torch.tensor(sent, dtype=torch.long, device='cuda:' + str(self.gpu_id))
            embedding = self.embeds(sent_tensor)
            embeddings.append(embedding.unsqueeze(dim=0))
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings
