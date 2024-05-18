import argparse
from process import Process

import warnings
warnings.filterwarnings('ignore')


# 设定参数列表
def get_args():
    parser = argparse.ArgumentParser(description='Script of mmWaver distinguish',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', default=42, help='number of random seed')
    parser.add_argument('--lr', default=0.0005, help='learning rate')
    parser.add_argument('--drop_prob', default=0.7, help='dropout rate')
    parser.add_argument('--batch_size', default=32, help='size of each batch')
    parser.add_argument('--max_epochs', default=5, help='max epochs of running model')
    parser.add_argument('--src_domain', default='bag', help='source domain')
    parser.add_argument('--tag_domain', default='fast', help='target domain')

    return parser.parse_args()


def main():
    args = get_args()
    Process(args).train()


if __name__ == "__main__":
    main()
