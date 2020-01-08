from models import sketchRNNmodel
import torch
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sketch Generation Model')

    parser.add_argument('--dist_matching', type=str, default='MMD')  # KL vs MMD
    parser.add_argument('--TransEncoder', default=True) # TransformerEncoder vs LSTM Encoder

    parser.add_argument('--enc_rnn_size', default=256)
    parser.add_argument('--dec_rnn_size', default=512)
    parser.add_argument('--z_size', default=128)

    parser.add_argument('--num_mixture', default=20)
    parser.add_argument('--input_dropout_prob', default=0.9)
    parser.add_argument('--output_dropout_prob', default=0.9)
    parser.add_argument('--batch_size', default=100)

    parser.add_argument('--kl_weight_start', default=0.01)
    parser.add_argument('--kl_decay_rate', default=0.99995)
    parser.add_argument('--kl_tolerance', default=0.2)
    parser.add_argument('--kl_weight', default=100)

    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--decay_rate', default=0.9999)
    parser.add_argument('--min_learning_rate', default=0.00001)
    parser.add_argument('--grad_clip', default=1.)

    parser.add_argument('--max_seq_len', default=100)
    parser.add_argument('--random_scale_factor', default=0.15)
    parser.add_argument('--augment_stroke_prob', default=0.10)
    parser.add_argument('--dim_feedforward', default=2048)
    parser.add_argument('--single_embedding', default=True)


    hp = parser.parse_args()
    print(hp)

    #####################################################
    ############## Sample Data ##########################

    model = sketchRNNmodel(hp)
    model.load_state_dict(torch.load('./model/model_birthday_cake_85000_.pth', map_location=device))

    generated_samples = model.generation(number_of_sample=25)

