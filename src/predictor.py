"""
    Predict with tarined model.
"""
import torch
import config
import get_logger
logger = get_logger.get_logger(name='predictor')
from visualization.save_images_gray_grid import save_images_grid


class Predictor():
    def __init__(self, model, writer, device,
                 window, gpu_id, use_rotate, max_predict):

        self.model = model
        self.writer = writer
        self.device = device
        self.window = window
        self.gpu_id = gpu_id
        self.use_rotate = use_rotate
        self.max_predict = max_predict

    def predict(self, test_loader):
        """Predict/Reconstruct Image from batch data x.
            Args:
                x (Tensor)                  : context. Binary Image (Rotation, Channel, Height, Width)
                epoch - global_step (int)   : Save result image by global_step.
                batch_idx (int)             : Save result image named by BATCH_[batc_idx]
        """
        for batch_idx, data_dic in enumerate(test_loader):
            if batch_idx >= self.max_predict + self.window:
                break

            data, current_idx = data_dic, 0
            for i in range(data.shape[0]):
                if sum(sum(sum(sum(sum(data[i]))))) == 0.:
                    current_idx = config.error_idx
                    break

            if current_idx == config.error_idx:
                logger.debug("Skip this batch beacuse window can't load data")
                continue
            else:
                target, context = data[:, 0], data[:, 1:]
                target = target.contiguous().view(target.shape[0] * target.shape[1], target.shape[2], target.shape[3], target.shape[4])
                target = target.to(self.device)
                #context = context.to(device)
                logger.debug(target.shape)

                """
                    if batch_idx % args.num_of_tensor_to_embed == args.window:
                        left_context_cat, right_context_cat, target_cat = context[0, 0], context[1, 0], target[0]
                    else:
                        left_context_cat = torch.cat([left_context_cat, context[0, 0]])
                        right_context_cat = torch.cat([right_context_cat, context[1, 0]])
                        target_cat = torch.cat([target_cat, target[0]])

                    if left_context_cat.shape[0] == args.num_of_tensor_to_embed:
                        context_cat = torch.stack([left_context_cat, right_context_cat])
                        context_cat = torch.unsqueeze(context_cat, dim=2)
                        target_cat = torch.unsqueeze(target_cat, dim=1)
                        trainer.predict(context_cat, epoch=0, batch_idx=batch_idx // args.num_of_tensor_to_embed)
                        del context_cat, target_cat
                """

                enc_x = self.model.encode(target)
                enc_x = self.model.m_original(enc_x)
                #enc_x = enc_x[:4]
                #target = target[:4]

                logger.debug("x:{}".format(enc_x.shape))

                self.writer.add_embedding(
                    mat=enc_x,
                    label_img=target,
                    global_step=batch_idx,
                    tag="test"
                    )

                save_images_grid(
                    target.cpu(),
                    nrow=config.nrow,
                    scale_each=True,
                    global_step=batch_idx,
                    tag_img="test/Output_data_batch{:0=3}".format(batch_idx),
                    writer=self.writer
                    )
