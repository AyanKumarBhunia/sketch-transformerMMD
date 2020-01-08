from Networks import *
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
from loss_functions import *
import torch.nn as nn
import numpy as np
import os



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

    def train_model(self, batch, lengths, step):
        self.train()
        self.optimizer.zero_grad()

        curr_learning_rate = ((self.hp.learning_rate - self.hp.min_learning_rate) *
                              (self.hp.decay_rate) ** step + self.hp.min_learning_rate)
        curr_kl_weight = (self.hp.kl_weight - (self.hp.kl_weight - self.hp.kl_weight_start) *
                          (self.hp.kl_decay_rate) ** step)

        post_dist = self.encoder(batch, lengths)


        z_vector = post_dist.rsample()
        start_token = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.hp.batch_size).unsqueeze(0).to(device)
        batch_init = torch.cat([start_token, batch], 0)
        z_stack = torch.stack([z_vector] * (self.hp.max_seq_len + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        output, _ = self.decoder(inputs, z_vector, lengths+1)

        end_token = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.shape[1]).unsqueeze(0).to(device)
        batch = torch.cat([batch, end_token], 0)
        x_target = batch.permute(1,0,2) #batch-> Seq_Len, Batch, Feature_dim


        #################### Loss Calculation ########################################
        ##############################################################################
        recons_loss = reconstruction_loss(output, x_target)

        if self.hp.dist_matching == 'KL':
            #################### KL Loss ########################################
            #####################################################################
            prior_distribution = torch.distributions.Normal(torch.zeros_like(post_dist.mean), torch.ones_like(post_dist.stddev))
            kl_cost =  torch.max(torch.distributions.kl_divergence(post_dist, prior_distribution).sum(),
                                                 torch.tensor(self.hp.kl_tolerance).to(device))
            loss = recons_loss + curr_kl_weight * kl_cost
            #####################################################################
        elif self.hp.dist_matching == 'MMD':
            z_fake = torch.randn(z_vector.shape).to(device)
            kl_cost = mmd_penalty(z_vector, z_fake)
            loss = recons_loss + 100 * kl_cost

        #################### Update Gradient ########################################
        #############################################################################
        self.set_learninRate(curr_learning_rate)
        loss.backward()
        nn.utils.clip_grad_norm(self.train_params, self.hp.grad_clip)
        self.optimizer.step()

        return kl_cost.item(), recons_loss.item(), loss.item(), curr_learning_rate, curr_kl_weight



    ##############################################################################
    ##############################################################################
    ###############Evaluation (Conditional or Random-Noise based) ################
    ##############################################################################

    def generation(self, dataloader, step, number_of_sample = 100, condition = False, foldername='Conditional'):

        Batch_Input = []
        Batch_Reconstructed = []
        row_count = 0
        col_count = 0

        for i_x in range(number_of_sample):
            self.eval()
            batch, lengths = dataloader.valid_batch(1)
            if condition:
                post_dist = self.encoder(batch, lengths)
                z_vector = post_dist.sample()
            else:
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
            batch_input = to_normal_strokes(batch[:,0,:].cpu().numpy())

            if (i_x + 0) % 10 == 0:
                row_count = row_count + 1
                col_count = 0

            Batch_Input.append([batch_input, [row_count-1, col_count]])
            Batch_Reconstructed.append([gen_strokes, [row_count-1, col_count]])
            col_count = col_count + 1
            print(i_x)

        Batch_Input_grid = make_grid_svg(Batch_Input)
        Batch_Reconstructed_grid = make_grid_svg(Batch_Reconstructed)
        if not os.path.exists(self.hp.foldername):
            os.makedirs(self.hp.foldername)
        if condition:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        draw_strokes(Batch_Input_grid, svg_filename= './' + self.hp.foldername + '/Input_'+ str(step) + 'sample.svg')
        draw_strokes(Batch_Reconstructed_grid, svg_filename= './' + self.hp.foldername  + '/Output_'+ str(step) + 'sample.svg')

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

    def set_learninRate(self, curr_learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = curr_learning_rate


    def draw_batch_input(self, dataloader, number_of_sample=100):


        Batch_Input = []
        row_count = 0
        col_count = 0

        for i_x in range(number_of_sample):
            batch, lengths = dataloader.valid_batch(1)
            batch_input = to_normal_strokes(batch[:, 0, :].cpu().numpy())

            if (i_x + 0) % 10 == 0:
                row_count = row_count + 1
                col_count = 0

            Batch_Input.append([batch_input, [row_count - 1, col_count]])
            col_count = col_count + 1
            print(i_x)

        Batch_Input_grid = make_grid_svg(Batch_Input)
        draw_strokes(Batch_Input_grid, svg_filename='sample.svg')



