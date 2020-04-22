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

    def fit(self, train_loader, test_loader):

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()

            loss_mean_epoch = 0

            logger.info("Epoch: %d/%d \tGPU: %d" % (epoch, self.max_epoch, int(self.gpu_id)))

            for batch_idx, data_dic in enumerate(train_loader):

                data_idx, data = self.get_data_from_dic(data_dic)

                if data_idx == config.error_idx:
                    continue
                else:
                    target, context = self.slice_data(self.use_rotate, data)
                    target, context = target.to(self.device), context.to(self.device)

                self.optimizer.zero_grad()

                recon_x, _, _ = self.model.forward(context)

                loss = self.model.loss_function(target, recon_x)

                loss.backward()

                self.optimizer.step()

                if batch_idx % (len(train_loader) // 20) == 0:
                    logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                        batch_idx, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

                self.writer.add_scalar(tag="train_loss_step_batch/loss_000",
                                       scalar_value=loss.item(), global_step=batch_idx)

                loss_mean_epoch += loss.item()

            self.writer.add_scalar(tag="train_loss_step_epoch/loss_000",
                                   scalar_value=loss_mean_epoch/len(train_loader.dataset), global_step=epoch)

            torch.save(self.model.state_dict(), "../models/__" + str(epoch) + ".pkl")

            self.evaluate(test_loader, epoch)

    def predict(self, x, target, epoch=0, batch_idx=0):
        """Predict/Reconstruct Image from batch data x.
            Args:
                x (Tensor)                  : context. Binary Image (Rotation, Channel, Height, Width)
                target (Tensor)             : target. Binary Image (Rotation, Channel, Height, Width)
                epoch - global_step (int)   : Save result image by global_step.
                batch_idx (int)             : Save result image named by BATCH_[batc_idx]
        """
        x.to(self.device)
        recon_x, _, _ = self.model.forward(x)

        save_images_grid(x.cpu(), nrow=6, scale_each=True, global_step=epoch,
                         tag_img="Input_data/BATCH_{0:0=3}".format(batch_idx), writer=self.writer)
        save_images_grid(target.cpu(), nrow=6, scale_each=True, global_step=epoch,
                         tag_img="Output_data/BATCH_{0:0=3}".format(batch_idx), writer=self.writer)
        save_images_grid(recon_x, nrow=6, scale_each=True, global_step=epoch,
                         tag_img="Reconstruct_from_data/BATCH_{0:0=3}".format(batch_idx), writer=self.writer)

    def evaluate(self, test_loader, epoch=0):
        """Evaluate model with test dataset.
            Args:
                test_loader (Dataset) :
                epoch (int)           :Save result image by global_step.
        """

        with torch.no_grad():
            self.model.eval()
            loss_mean_epoch = 0

            for batch_idx, data_dic in enumerate(test_loader):

                data_idx, data = self.get_data_from_dic(data_dic)

                if data_idx == config.error_idx:
                    continue
                else:
                    target, context = self.slice_data(self.use_rotate, data)
                    target, context = target.to(self.device), context.to(self.device)

                recon_x, _, _ = self.model.forward(context)

                loss = self.model.loss_function(target, recon_x)

                if batch_idx % (len(test_loader) // 10) == 0:
                    logger.debug("Eval batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                                 batch_idx, len(test_loader.dataset),
                                 100. * batch_idx / len(test_loader), loss.item()))

                self.writer.add_scalar(tag="eval_loss_step_batch/loss_000",
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
    def slice_data(use_rotate, data):
        """Slice data, get target, context.
            Args:
                use_rotate  :(Batchsize, ContextOrTarget, Rotation, C, H, W) into (R, C, H, W)
                not         :(Batchsize, ContextOrTarget, Rotation, C, H, W) into (1, C, H, W)
            Return:
                target (Tensor)   :Image @ t[sec]
                context (Tensor)  :Image @ t-w, ..., t-1, t+1, ..., t+w[sec]
        """
        if use_rotate:
            target, context = data[0, 0], data[0, 1]
        else:
            target, context = data[:, 0, 0], data[:, 1, 0]
        return target, context
