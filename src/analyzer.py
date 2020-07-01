"""
    Analyze tarined model.
"""
import torch
import torch.nn as nn
import config
import get_logger
logger = get_logger.get_logger(name='analyzer')
from models.worm2vec_non_sequential import Worm2vec_nonseq
from visualization.save_images_gray_grid import save_images_grid
from trainer import Trainer

class Analyzer():
    def __init__(self, model, writer, device,
                 gpu_id, max_analyze):

        self.model = model
        self.writer = writer
        self.device = device
        self.gpu_id = gpu_id
        self.max_analyze = max_analyze
        self.redefined_model = ReDefined_Worm2vec_non_seq(self.model)

    def analyze(self, test_loader):
        """Analyze model to access the output of middle layers

        Args:
            test_loader ([type]): [description]
        """
        with torch.no_grad():

            for batch_idx, data in enumerate(test_loader):
                if batch_idx >= self.max_analyze:
                    break

                target, labels = Trainer.slice_data(data)
                target = target.to(self.device)
                #labels = labels.to(device)
                logger.debug(target.shape)

                results = self.redefined_model.forward(target)

                logger.debug("results:{}".format(len(results), results[0].shape))

                target_i = target[:4].cpu()
                save_images_grid(
                    target_i,
                    nrow=config.nrow,
                    scale_each=True,
                    global_step=batch_idx,
                    tag_img="Input",
                    writer=self.writer
                    )
                for layer_i, result in enumerate(results):
                    for channel in range(result.shape[1]):
                        result_i = result[:4, channel].unsqueeze(1).cpu()
                        save_images_grid(
                            result_i,
                            nrow=config.nrow,
                            scale_each=True,
                            global_step=batch_idx,
                            tag_img="Output_of_midium_layer_{:0=2}/channel_{:0=2}".format(layer_i, channel),
                            writer=self.writer
                            )

class ReDefined_Worm2vec_non_seq(object):
    def __init__(self, trained_model):
        super(ReDefined_Worm2vec_non_seq, self).__init__()
        features = list(trained_model.enc)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if type(model) is type(nn.ReLU()):
                results.append(x)
        return results
