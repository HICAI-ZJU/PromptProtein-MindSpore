import json
import os
import pickle
from argparse import ArgumentParser, Namespace
import argparse
import mindspore as ms


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--emb_layer_norm_before', type=bool,
                        default= 'True')
    parser.add_argument('--token_dropout', type=bool,
                        default= 'False')
    parser.add_argument('--num_layers', type=int,
                        default= '33')
    parser.add_argument('--embed_dim', type=int,
                        default='1280')
    parser.add_argument('--ffn_embed_dim', type=int,
                        default='5120')
    parser.add_argument('--attention_heads', type=int,
                        default='20')
    parser.add_argument('--max_positions', type=int,
                        default='1024')



def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args

