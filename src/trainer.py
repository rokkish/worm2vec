"""
    Trainer model.
"""
import torch
import numpy
import config
import get_logger
logger = get_logger.get_logger(name='trainer')
from visualization.save_images_gray_grid import save_images_grid

class Trainer():
    def __init__(self, model, optimizer, writer, device, \
        epoch, window, gpu_id, use_rotate):

        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.max_epoch = epoch
        self.window = window
        self.gpu_id = gpu_id
        self.use_rotate = use_rotate
        self.current_epoch = 0

    def fit(self, train_loader):

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()

            logger.info("Epoch: %d/%d \tGPU: %d" % (epoch, self.max_epoch, int(self.gpu_id)))

            for batch_idx, data_dic in enumerate(train_loader):

                data_idx, data = self.get_data_from_dic(data_dic)

                if data_idx == config.error_idx:
                    logger.debug("Skip this batch beacuse window can't load data")
                    continue
                else:
                    if self.use_rotate:
                        logger.debug(data.shape)
                        target, context = data[0, 0], data[0, 1]
                    else:
                        target, context = data[:, 0, 0], data[:, 1, 0]
                    target, context = target.to(self.device), context.to(self.device)

                self.optimizer.zero_grad()

                recon_x, _, _ = self.model.forward(context)

                loss = self.model.loss_function(target, recon_x)

                loss.backward()

                self.optimizer.step()

                if batch_idx % (len(train_loader) // 10) == 0:
                    logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                            batch_idx * len(target), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                
                self.writer.add_scalar(tag="train_loss_step_batch/loss_000", scalar_value=loss.item(), global_step=batch_idx)

            self.writer.add_scalar(tag="train_loss_step_epoch/loss_000", scalar_value=loss.item(), global_step=epoch)

    @staticmethod
    def get_data_from_dic(data_dic):
        """
        data_dic = {
            data_idx(int):data(tensor)
            }
        """
        for k, v in data_dic.items():
            data_idx, data = k, v
        return data_idx, data
    