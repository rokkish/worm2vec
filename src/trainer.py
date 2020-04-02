"""
    Trainer model.
"""
import torch
import numpy
import get_logger
logger = get_logger.get_logger(name='trainer')

class Trainer():
    def __init__(self, model, optimizer, writer, device):
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.current_epoch = 0

    def fit(self, train_loader, args):
        model, optimizer = self.model, self.optimizer
        writer, device = self.writer, self.device

        for epoch in range(1, args.epoch + 1):
            model.train()

            logger.info("Epoch: %d/%d \tGPU: %d" % (epoch, args.epoch, int(args.gpu_id)))

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                loss = model.forward(data, target)

                model.backward()

                optimizer.step()

                if batch_idx % (len(train_loader) // 10) == 0:
                    logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                            batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))

                writer.add_scalar(tag="train_loss_step_batch/loss_000", scalar_value=loss.item(), global_step=batch_idx)

            writer.add_scalar(tag="train_loss_step_epoch/loss_000", scalar_value=loss.item(), global_step=epoch)
