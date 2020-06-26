"""
    Trainer model.
"""
import torch
import config
import get_logger
logger = get_logger.get_logger(name='trainer')
from visualization.save_images_gray_grid import save_images_grid


class Trainer():
    def __init__(self, model, optimizer, writer, device,
                 epoch, gpu_id):

        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.max_epoch = epoch
        self.gpu_id = gpu_id

    def fit(self, train_loader, test_loader):

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()

            loss_mean_epoch = 0

            logger.info("Epoch: %d/%d GPU: %d" % (epoch, self.max_epoch, int(self.gpu_id)))

            for batch_idx, data in enumerate(train_loader):

                target = self.slice_data(data)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                #logger.debug("context: %s, target: %s" % (context.shape, target.shape))

                loss = self.model.forward(target)

                loss.backward()

                self.optimizer.step()

                if batch_idx % (len(train_loader) // 10) == 0:
                    logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                        batch_idx * config.BATCH_SIZE, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

                self.writer.add_scalar(tag="train_loss_step_batch/loss_{:0=3}".format(epoch),
                                       scalar_value=loss.item(), global_step=batch_idx)

                loss_mean_epoch += loss.item()

            self.writer.add_scalar(tag="train_loss_step_epoch/loss_000",
                                   scalar_value=loss_mean_epoch/len(train_loader.dataset), global_step=epoch)

            torch.save(self.model.state_dict(), "../models/__" + str(epoch) + ".pkl")

            logger.debug("TRAIN STOP, EVAL START")
            self.evaluate(test_loader, epoch)
            logger.debug("EVAL STOP, ")


    def evaluate(self, test_loader, epoch=0):
        """Evaluate model with test dataset.
            Args:
                test_loader (Dataset) :
                epoch (int)           :Save result image by global_step.
        """

        with torch.no_grad():
            self.model.eval()
            loss_mean_epoch = 0

            for batch_idx, data in enumerate(test_loader):

                target = self.slice_data(data)
                target = target.to(self.device)

                loss = self.model.forward(target)

                if batch_idx % (len(test_loader) // 10) == 0:
                    logger.debug("Eval batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                                 batch_idx * config.BATCH_SIZE, len(test_loader.dataset),
                                 100. * batch_idx / len(test_loader), loss.item()))

                self.writer.add_scalar(tag="eval_loss_step_batch/loss_{:0=3}".format(epoch),
                                       scalar_value=loss.item(), global_step=batch_idx)

                #if batch_idx > config.MAX_LEN_EVA_LDATA:
                #    break
                loss_mean_epoch += loss.item()

            self.writer.add_scalar(tag="eval_loss_step_epoch/loss_000",
                                   scalar_value=loss_mean_epoch/len(test_loader.dataset), global_step=epoch)

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

    @staticmethod
    def slice_data(data):
        """Slice data, get target
        Args:
            data (tensor)   : (Batchsize, Rotation, C, H, W)
        Return:
            target (Tensor) : (B, C, H, W)
        """
        target = data[:, 0]
        target = target.contiguous().view(target.shape[0]*target.shape[1], target.shape[2], target.shape[3], target.shape[4])
        return target
