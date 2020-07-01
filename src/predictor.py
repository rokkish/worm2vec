"""
    Predict with tarined model.
"""
import torch
import config
import get_logger
logger = get_logger.get_logger(name='predictor')
from visualization.save_images_gray_grid import save_images_grid
from trainer import Trainer

class Predictor():
    def __init__(self, model, writer, device,
                 gpu_id, max_predict):

        self.model = model
        self.writer = writer
        self.device = device
        self.gpu_id = gpu_id
        self.max_predict = max_predict

    def predict(self, test_loader):
        """Predict/Reconstruct Image from batch data x.
            Args:
                x (Tensor)                  : context. Binary Image (Rotation, Channel, Height, Width)
                epoch - global_step (int)   : Save result image by global_step.
                batch_idx (int)             : Save result image named by BATCH_[batc_idx]
        """
        for batch_idx, data in enumerate(test_loader):
            if batch_idx >= self.max_predict:
                break

            target, labels = Trainer.slice_data(data)
            target = target.to(self.device)
            #labels = labels.to(device)
            logger.debug(target.shape)

            enc_x = self.model.encode(target)
            clas_x = self.model.classifier(enc_x)
            target = target[:4]
            enc_x = enc_x[:4]
            clas_x = clas_x[:4]

            logger.debug("x:{}".format(enc_x.shape))

            self.writer.add_embedding(
                mat=enc_x,
                label_img=target,
                global_step=batch_idx,
                tag="test_embedd"
                )
            self.writer.add_embedding(
                mat=clas_x,
                label_img=target,
                global_step=batch_idx,
                tag="test_class"
                )

            save_images_grid(
                target.cpu(),
                nrow=config.nrow,
                scale_each=True,
                global_step=batch_idx,
                tag_img="test/Output_data_batch{:0=3}".format(batch_idx),
                writer=self.writer
                )
