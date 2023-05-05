import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn import MSELoss

from .dataset import StockDataset, TestDataset
from .model.ns_transformer import NSTransformer
from .utils import *


def train(config: Config, logger):
    train_dataset = StockDataset(config, "train")
    valid_dataset = StockDataset(Config, "valid")
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)

    path = os.path.join(config.model_save_path, "")
    if not os.path.exists(path):
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    model = NSTransformer(config).to(device)

    if config.add_train:
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1,
                                      total_iters=int(train_steps * config.train_epochs * 0.8))
    criterion = MSELoss()

    for epoch in range(config.train_epochs):
        logger.info("Epoch {}/{}".format(epoch, config.train_epochs))
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(device)

            # encoder - decoder
            if config.output_attention:
                outputs = model(batch_x, dec_inp)[0]
            else:
                outputs = model(batch_x, dec_inp)
            outputs = outputs[:, -config.pred_len:, :]
            batch_y = batch_y[:, -config.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f} | speed: {3:.4f}s/iter".format(i + 1, epoch + 1,
                                                                                                     loss.item(),
                                                                                                     speed))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(model, valid_loader, criterion, config)

        logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, path, logger, config.model_name)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    best_model_path = path + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model


def vali(model, vali_loader, criterion, config):
    total_loss = []
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            if config.output_attention:
                outputs = model(batch_x, dec_inp)[0]
            else:
                outputs = model(batch_x, dec_inp)
            outputs = outputs[:, -config.pred_len:, :]
            batch_y = batch_y[:, -config.pred_len:, :].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def predict(config):
    test_dataset = TestDataset(Config)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = NSTransformer(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    result = torch.Tensor().to(device)
    label = torch.Tensor().to(device)

    model.eval()
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(device)
        # encoder - decoder
        if config.output_attention:
            outputs = model(batch_x, dec_inp)[0]
        else:
            outputs = model(batch_x, dec_inp)
        outputs = outputs[:, -config.pred_len:, :]
        batch_y = batch_y[:, -config.pred_len:, :].to(device)

        pred = outputs.detach()
        true = batch_y.detach()
        result = torch.cat((result, pred), dim=0)
        label = torch.cat((label, true), dim=0)

    return result.cpu().numpy(), label.cpu().numpy(), test_dataset.data_provider.data
