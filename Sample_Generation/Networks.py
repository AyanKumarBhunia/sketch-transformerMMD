import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *


class EncoderRNN(nn.Module):
    def __init__(self, hp):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(5, hp.enc_rnn_size, dropout=hp.input_dropout_prob, bidirectional=True)
        self.fc_mu = nn.Linear(2*hp.enc_rnn_size, hp.z_size)
        self.fc_sigma = nn.Linear(2*hp.enc_rnn_size, hp.z_size)

    def forward(self, x, Seq_Len=None):
        x = pack_padded_sequence(x, Seq_Len, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x.float())
        h_n = h_n.permute(1,0,2).reshape(h_n.shape[1], -1)
        mean = self.fc_mu(h_n)
        log_var = self.fc_sigma(h_n)
        posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        return posterior_dist

class EncoderTrans(nn.Module):
    def __init__(self, hp, dmodel=512):
        super(EncoderTrans, self).__init__()
        self.hp = hp
        self.src_mask = None

        if hp.single_embedding:
            self.emb = nn.Linear(5, dmodel)
        else:
            self.emb_1 = nn.Linear(2, int(3*(dmodel/4)))
            self.emb_2 = nn.Embedding(3, int((dmodel / 4)))
        self.pos_encoder = PositionalEncoding(dmodel)
        encoder_layers = nn.TransformerEncoderLayer(dmodel, 4, hp.dim_feedforward)
        encoder_norm = torch.nn.LayerNorm(dmodel)
        self.encoder = torch.nn.TransformerEncoder(encoder_layers, 2, encoder_norm)
        self.fc_mu = nn.Linear(dmodel, hp.z_size)
        self.fc_sigma = nn.Linear(dmodel, hp.z_size)


    def forward(self, x, Seq_Len=None):
        src_key_pad_mask = torch.zeros(x.shape[1], x.shape[0])
        for i_k, seq in enumerate(Seq_Len):
            src_key_pad_mask[i_k, seq:] = 1
        src_key_pad_mask= src_key_pad_mask.type(torch.bool)

        if self.hp.single_embedding:
            x = self.emb(x.permute(1, 0, 2)).permute(1, 0, 2)
        else:
            x_1 = self.emb_1(x.permute(1, 0, 2)[:,:, :2]).permute(1, 0, 2)
            x_2 = self.emb_2(x.permute(1, 0, 2)[:,:, 2:].long().argmax(-1)).permute(1, 0, 2)
            x = torch.cat((x_1, x_2), dim=-1)
        x = self.encoder(self.pos_encoder(x), src_key_padding_mask=src_key_pad_mask.to(device))
        last_time_step = []
        for i_k, seq in enumerate(Seq_Len):
            last_time_step.append(torch.max(x[:seq, i_k, :], dim=0)[0])
        last_time_step = torch.stack(last_time_step, dim=0)
        mean = self.fc_mu(last_time_step)
        log_var = self.fc_sigma(last_time_step)
        posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        return posterior_dist

class DecoderRNN(nn.Module):
    def __init__(self, hp):
        super(DecoderRNN, self).__init__()
        self.fc_hc = nn.Linear(hp.z_size, 2 * hp.dec_rnn_size)
        self.lstm = nn.LSTM(hp.z_size + 5, hp.dec_rnn_size, dropout=hp.output_dropout_prob)
        self.fc_params = nn.Linear(hp.dec_rnn_size, 6 * hp.num_mixture + 3)
        self.hp = hp

    def forward(self, inputs, z_vector, seq_len = None, hidden_cell=None, isTrain = True):
        self.training = isTrain
        if hidden_cell is None:
            hidden, cell = torch.split(F.tanh(self.fc_hc(z_vector)), self.hp.dec_rnn_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())

        if seq_len is None:
            seq_len = torch.tensor([1]).type(torch.int64).to(device)

        inputs = pack_padded_sequence(inputs, seq_len, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        outputs, _ = pad_packed_sequence(outputs)

        if self.training:
            if outputs.shape[0] != (self.hp.max_seq_len + 1):
                pad = torch.zeros(outputs.shape[-1]).repeat(self.hp.max_seq_len + 1 - outputs.shape[0], 100, 1).to(device)
                outputs = torch.cat((outputs, pad), dim=0)
            y_output = self.fc_params(outputs.permute(1,0,2))
        else:
            y_output = self.fc_params(hidden.permute(1,0,2))

        z_pen_logits = y_output[:, :, 0:3]
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y_output[:, :, 3:], 6, 2)
        z_pi = F.softmax(z_pi, dim=-1)
        z_sigma1 = torch.exp(z_sigma1)
        z_sigma2 = torch.exp(z_sigma2)
        z_corr = torch.tanh(z_corr)

        return [z_pi.reshape(-1, 20), z_mu1.reshape(-1, 20), z_mu2.reshape(-1, 20), \
               z_sigma1.reshape(-1, 20), z_sigma2.reshape(-1, 20), z_corr.reshape(-1, 20), z_pen_logits.reshape(-1, 3)], (hidden, cell)
