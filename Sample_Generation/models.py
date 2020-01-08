from Networks import *
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
import torch.nn as nn
import numpy as np
import math

class sketchRNNmodel(nn.Module):
    def __init__(self, hp):
        super(sketchRNNmodel, self).__init__()
        if hp.TransEncoder:
            self.encoder = EncoderTrans(hp).to(device)
        else:
            self.encoder = EncoderRNN(hp).to(device)
        self.decoder = DecoderRNN(hp).to(device)
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.hp = hp

    ##############################################################################
    ##############################################################################
    ###############Evaluation (Conditional or Random-Noise based) ################
    ##############################################################################

    def generation(self, number_of_sample = 100):
        Batch_Reconstructed = []
        row_count = 0
        col_count = 0

        for i_x in range(number_of_sample):
            self.eval()
            z_vector = torch.randn(1,128).to(device)
            start_token = torch.Tensor([0,0,1,0,0]).view(1,1,-1).to(device)
            state = start_token
            hidden_cell = None
            gen_strokes = []
            for i in range(self.hp.max_seq_len):
                input = torch.cat([state, z_vector.unsqueeze(0)],2)
                output, hidden_cell = self.decoder(input, z_vector, hidden_cell = hidden_cell, isTrain = False)
                state, next_state = self.sample_next_state(output)
                gen_strokes.append(next_state)

            gen_strokes = torch.stack(gen_strokes).cpu().numpy()
            gen_strokes = to_normal_strokes(gen_strokes)
            # gen_strokes is the every generated sample.

            if (i_x + 0) % math.sqrt(number_of_sample) == 0:
                row_count = row_count + 1
                col_count = 0

            Batch_Reconstructed.append([gen_strokes, [row_count-1, col_count]])
            col_count = col_count + 1
            print(i_x)

        Batch_Reconstructed_grid = make_grid_svg(Batch_Reconstructed)
        draw_strokes(Batch_Reconstructed_grid,
                         svg_filename='sample.svg')

        return Batch_Reconstructed

    ##############################################################################
    ##############################################################################

    def sample_next_state(self, output, temperature =0.2):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = output
        # get mixture indices:
        o_pi = o_pi.data[0,:].cpu().numpy()
        o_pi = adjust_temp(o_pi)
        pi_idx = np.random.choice(self.hp.num_mixture, p=o_pi)
        # get pen state:
        o_pen = F.softmax(o_pen_logits, dim=-1)
        o_pen = o_pen.data[0,:].cpu().numpy()
        pen = adjust_temp(o_pen)
        pen_idx = np.random.choice(3, p=pen)
        # get mixture params:
        o_mu1 = o_mu1.data[0,pi_idx].item()
        o_mu2 = o_mu2.data[0,pi_idx].item()
        o_sigma1 = o_sigma1.data[0,pi_idx].item()
        o_sigma2 = o_sigma2.data[0,pi_idx].item()
        o_corr = o_corr.data[0,pi_idx].item()
        x,y = sample_bivariate_normal(o_mu1,o_mu2,o_sigma1,o_sigma2,o_corr, temperature = temperature, greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[pen_idx+2] = 1
        return next_state.to(device).view(1,1,-1), next_state


