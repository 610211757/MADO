# coding:utf-8

# import click
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='kdd99')
    parser.add_argument("--v", type=int, default=1000)

    parser.add_argument("--data_dir", type=str, default='dataset', help="path of dataset")
    parser.add_argument("--processed_data_dir", type=str, default='processed_data_dir')

    parser.add_argument("--log_dir", type=str, default='log', help="The directory to store the res")

    parser.add_argument("--usingPCA", type=bool, default=False)
    # parser.add_argument("--using_stat_features", type=bool, default=False)

    parser.add_argument("--seq_length", type=int, default=120)
    parser.add_argument("--seq_step", type=int, default=10)

    parser.add_argument("--seq_swat_length", type=int, default=120)
    parser.add_argument("--seq_swat_step", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=1)

    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--gcn_hidden_dim", type=int, default=64)
    parser.add_argument("--gcn_out_dim", type=int, default=16)
    parser.add_argument("--cnn_out_dim", type=int, default=64)
    parser.add_argument("--global_out_dim", type=int, default=64)
    parser.add_argument("--local_out_dim", type=int, default=32)
    parser.add_argument("--final_out_dim", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--nums_lstm_layer", type=int, default=1)

    parser.add_argument("--mode", type=str,
                        # type=click.Choice(["lstm-cnn-gcn-one-class", "lstm-cnn-one-class", "lstm-one-class"]),
                        default="lstm-cnn-gcn-one-class")
    
    parser.add_argument("--add_lstm_prediction", type=bool, default=True)
    parser.add_argument("--weight_lstm_prediction", type=float, default=0.001)

    parser.add_argument("--allow_gcn_to_lstm", type=bool, default=True)
    parser.add_argument("--add_learned_r", type=bool, default=True)

    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--load_model", type=bool, default=False)
    
    args = parser.parse_args()
    return args



