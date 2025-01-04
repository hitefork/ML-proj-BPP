import argparse


parser = argparse.ArgumentParser(description='PyTorch Soft Actor Critic')

parser.add_argument('--episodes', type=int, default=100001,
                    help='number of episodes to train the agent')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=43, metavar='N',
                    help='random seed (default: 43)')
parser.add_argument('--save_path', type=str, default='./Models/', 
                    help='file path to save the weights')
parser.add_argument('--load_path', type=str, default=None, 
                    help='path to load model from pre-trained weights')
parser.add_argument('--tensorboard', type=int, default=1, 
                    help='Whether we want tensorboardX logging')
parser.add_argument('--batch_size', type=int, default=20, 
                    help='batch size to sample')




args = parser.parse_args()