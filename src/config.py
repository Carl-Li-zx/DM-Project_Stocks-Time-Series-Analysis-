import os
import time


class Config:
    feature_columns = [2, 3, 4, 5, 6]  # 要作为feature的列

    pred_len = 4
    seq_len = 16
    label_len = 8

    output_attention = False
    enc_in = len(feature_columns)
    dec_in = enc_in
    c_out = dec_in
    d_model = 8
    d_ff = 4 * d_model
    n_heads = 4

    e_layers = 3
    d_layers = 3

    p_hidden_dims = (16, 16)
    p_hidden_layers = 2

    activation = "gelu"

    dropout = 0.1

    do_train = True
    do_predict = True
    add_train = False
    use_cuda = True

    train_data_rate = 0.9
    batch_size = 64
    learning_rate = 1e-5
    train_epochs = 40
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



