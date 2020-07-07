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
                 epoch, gpu_id, loss_function):

        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.max_epoch = epoch
        self.gpu_id = gpu_id
        self.loss_function = loss_function

    def fit(self, train_loader, test_loader):

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()

            loss_mean_epoch = 0

            logger.info("Epoch: %d/%d GPU: %d" % (epoch, self.max_epoch, int(self.gpu_id)))

            for batch_idx, (anchor, positive, negative) in enumerate(train_loader):

                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                anchor = anchor.view(anchor.shape[0]*anchor.shape[1], anchor.shape[2], anchor.shape[3], anchor.shape[4])
                positive = positive.view(positive.shape[0]*positive.shape[1], positive.shape[2], positive.shape[3], positive.shape[4])
                negative = negative.view(negative.shape[0]*negative.shape[1], negative.shape[2], negative.shape[3], negative.shape[4])

                self.optimizer.zero_grad()

                #logger.debug("anc: %s, pos: %s, neg:%s" % (anchor.shape, positive.shape, negative.shape))

                output = self.model.forward(anchor, positive, negative)
                loss = self.loss_function(output)

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

            for batch_idx, (anchor, positive, negative) in enumerate(test_loader):

                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                anchor = anchor.view(anchor.shape[0]*anchor.shape[1], anchor.shape[2], anchor.shape[3], anchor.shape[4])
                positive = positive.view(positive.shape[0]*positive.shape[1], positive.shape[2], positive.shape[3], positive.shape[4])
                negative = negative.view(negative.shape[0]*negative.shape[1], negative.shape[2], negative.shape[3], negative.shape[4])

                output = self.model.forward(anchor, positive, negative)
                loss = self.loss_function(output)

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
