"""
    Trainer model.
"""
import torch
import numpy
import get_logger
logger = get_logger.get_logger(name='trainer')

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

            for batch_idx, data in enumerate(train_loader):

                logger.debug("batch_idx:%d, trian_loader.dataset:%s, train_loader:%s" \
                 % (batch_idx, len(train_loader.dataset), len(train_loader)))

                if batch_idx - self.window < 0 or \
                    batch_idx + self.window > len(train_loader.dataset):
                    logger.debug("Skip this batch beacuse window can't load data")
                    continue
                else:
                    if self.use_rotate:
                        target, context = data[:, 0], data[:, 1:]
                    else:
                        target, context = data[:, 0, 0], data[:, 1:, 0]
                    target, context = target.to(self.device), context.to(self.device)
                    logger.debug("target:%s, context:%s" % (target.shape, context.shape))


                optimizer.zero_grad()

                loss = self.model.forward(target, context)

                self.model.backward()

                optimizer.step()

                if batch_idx % (len(train_loader) // 10) == 0:
                    logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                            batch_idx * len(target), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                
                writer.add_scalar(tag="train_loss_step_batch/loss_000", scalar_value=loss.item(), global_step=batch_idx)

            writer.add_scalar(tag="train_loss_step_epoch/loss_000", scalar_value=loss.item(), global_step=epoch)
