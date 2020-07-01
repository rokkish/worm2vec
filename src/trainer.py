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

            correct = 0
            total = 0

            for batch_idx, data in enumerate(train_loader):

                data, labels = self.slice_data(data)
                data, labels = data.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                #logger.debug("data: %s, labels: %s" % (data.shape, len(labels)))

                output = self.model.forward(data)

                #acc
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss_function(output, labels)

                loss.backward()

                self.optimizer.step()

                if batch_idx % (len(train_loader) // 10) == 0:
                    logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                        batch_idx * config.BATCH_SIZE, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

                self.writer.add_scalar(tag="train_loss_step_batch/loss_{:0=3}".format(epoch),
                                       scalar_value=loss.item(), global_step=batch_idx)

                loss_mean_epoch += loss.item()

            self.writer.add_scalar(tag="train_loss_step_epoch/CrossEntropyLoss",
                                   scalar_value=loss_mean_epoch/len(train_loader.dataset), global_step=epoch)
            self.writer.add_scalar(tag="train_loss_step_epoch/Acc",
                                   scalar_value=100 * correct / total, global_step=epoch)

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), "../models/__" + str(epoch) + ".pkl")

            logger.info("TRAIN: Accuracy: {:.3f}".format(100 * correct / total))
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
            total = 0
            correct = 0

            for batch_idx, data in enumerate(test_loader):

                data, labels = self.slice_data(data)
                data, labels = data.to(self.device), labels.to(self.device)

                output = self.model.forward(data)

                #acc
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.loss_function(output, labels)
                #logger.debug("_{}, pred:{}".format(_.shape, predicted.shape))
        
                if batch_idx % (len(test_loader) // 10) == 0:
                    logger.debug("Eval batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss: {:.5f}".format(
                                 batch_idx * config.BATCH_SIZE, len(test_loader.dataset),
                                 100. * batch_idx / len(test_loader), loss.item()))

                self.writer.add_scalar(tag="eval_loss_step_batch/loss_{:0=3}".format(epoch),
                                       scalar_value=loss.item(), global_step=batch_idx)

                #if batch_idx > config.MAX_LEN_EVA_LDATA:
                #    break
                loss_mean_epoch += loss.item()

            self.writer.add_scalar(tag="eval_loss_step_epoch/CrossEntropyLoss",
                                   scalar_value=loss_mean_epoch/len(test_loader.dataset), global_step=epoch)
            self.writer.add_scalar(tag="eval_loss_step_epoch/Acc",
                                   scalar_value=100 * correct / total, global_step=epoch)

            logger.info(" EVAL: Accuracy: {:.3f}".format(100 * correct / total))

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
            data (dic)   : target:(Batchsize, 1, C, H, W), rotation:(Batchsize, Rotation, C, H, W)
        Return:
            data    : original, rotation : (B, R+1, C, H, W)
            #target (Tensor)     : (B, 1, C, H, W)
            #rotation (Tensor)   : (B, R, C, H, W)
            labels  : onehot vector      : (B, R+1)
        """
        #target, rotation = data["target"], data["rotation"]
        #return target, rotation
        data, labels = data["data"], data["labels"]
        data = data.contiguous().view(data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4])
        labels = labels.view(labels.shape[0]*labels.shape[1])
        return data, labels