from models import sketchRNNmodel
from dataset import sketchRNN_Data
import torch
##################### hyperparameters #####################
class HParams():
    def __init__(self):
        self.data_set = 'cat.npz'
        self.enc_rnn_size = 256
        self.dec_rnn_size = 512
        self.z_size = 128
        self.num_mixture = 20
        self.input_dropout_prob = 0.9
        self.output_dropout_prob = 0.9
        self.batch_size = 100
        self.kl_weight_start = 0.01
        self.kl_decay_rate = 0.99995
        self.kl_tolerance = 0.2
        self.kl_weight = 100
        self.learning_rate = 0.001
        self.decay_rate = 0.9999
        self.min_learning_rate = 0.00001
        self.grad_clip = 1.
        self.max_seq_len = 200
        self.random_scale_factor = 0.15
        self.augment_stroke_prob = 0.10
        self.TransEncoder = True
        self.foldername = 'TransEncoder'
        self.dim_feedforward = 2048
        self.dist_matching = 'MMD' # KL vs MMD

        # self.num_steps = 100000  # Total number of steps of training. Keep large.
        # self.save_every = 5000  # Number of batches per checkpoint creation.
        # self.use_input_dropout = False  # Input dropout. Recommend leaving False.
        # self.use_output_dropout = False  # Output dropout. Recommend leaving False.
        # self.random_scale_factor = 0.15  # Random scaling data augmentation proportion.
        # self.augment_stroke_prob = 0.10  # Point dropping augmentation proportion.
        # self.conditional = True  # When False, use unconditional decoder-only model.
        # self.is_training = True  # Is model training? Recommend keeping true.

####################################################################################
####################################################################################

if __name__ == "__main__":
    hp = HParams()
    dataloader = sketchRNN_Data(hp)
    print(hp.foldername)

    #######################################################################
    ############################## End Load Data ##########################

    model = sketchRNNmodel(hp)
    for step in range(100001):
        batch_data, batch_len = dataloader.train_batch()
        kl_cost, recons_loss, loss, curr_learning_rate, curr_kl_weight  = model.train_model(batch_data, batch_len, step)

        print('Step:{} ** Current_LR:{} ** Current_KL:{} ** KL_Loss:{} '
              '** Recons_Loss:{} ** Total_loss:{}'.format(step, curr_learning_rate, curr_kl_weight,
                                                          kl_cost, recons_loss, loss ))
        if (step + 1) % 5000 == 0:
            model.generation(dataloader, step, number_of_sample=100, condition=False)
            #model.generation(dataloader, step, condition=True, foldername='Conditional')

        if (step + 1) % 5000 == 0:
            torch.save(model.state_dict(), 'sketchRNN_d_' + str(step) + '_.pth')

