import os
import time


class Config:
    feature_columns = [0, 1, 2, 3, 4]  # 要作为feature的列

    pred_len = 2
    seq_len = 24
    label_len = 16

    output_attention = False
    enc_in = len(feature_columns)
    dec_in = enc_in
    c_out = dec_in
    d_model = 16
    d_ff = 4 * d_model
    n_heads = 8

    e_layers = 8
    d_layers = 8

    p_hidden_dims = (16, 16)
    p_hidden_layers = 2

    activation = "gelu"

    dropout = 0.1

    do_train = True
    add_train = False
    use_cuda = True

    train_data_rate = 0.8
    batch_size = 1024
    learning_rate = 5e-5
    train_epochs = 60
    random_seed = 42
    patience = 7

    model_name = "ns_transformer.pth"

    train_data_path = "dataset/"
    model_save_path = "checkpoint/"
    figure_save_path = "figure/"
    log_save_path = "log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = True
    do_train_visualized = False
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)
