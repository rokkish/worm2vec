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
        for batch_idx, (anchor, positive, negative) in enumerate(test_loader):
            if batch_idx >= self.max_predict:
                break

            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
            anchor = anchor.view(anchor.shape[0]*anchor.shape[1], anchor.shape[2], anchor.shape[3], anchor.shape[4])
            positive = positive.view(positive.shape[0]*positive.shape[1], positive.shape[2], positive.shape[3], positive.shape[4])
            negative = negative.view(negative.shape[0]*negative.shape[1], negative.shape[2], negative.shape[3], negative.shape[4])


            enc_x = self.model.forward(anchor, positive, negative)
            anc_embedding = enc_x["anc_embedding"]
            pos_embedding = enc_x["pos_embedding"]
            neg_embedding = enc_x["neg_embedding"]


            cat_input = torch.cat([anchor, positive, negative])
            cat_embedding = torch.cat([anc_embedding, pos_embedding, neg_embedding])
            #enc_x = enc_x[:4]
            #target = target[:4]

            cat_input_reverse = torch.abs(cat_input - torch.ones(cat_input.shape).float().to(self.device))

            labels = ["anchor"]*len(anchor) + ["positive"]*len(positive) + ["negative"]*len(negative)
            time = list(range(0, cat_input.shape[0]))
            all_labels = list(zip(time, labels))


            self.writer.add_embedding(
                mat=cat_embedding,
                metadata=all_labels,
                metadata_header=["time", "type"],
                label_img=cat_input_reverse,
                global_step=batch_idx,
                tag="test"
                )

            save_images_grid(
                cat_input.cpu(),
                nrow=config.nrow,
                scale_each=True,
                global_step=batch_idx,
                tag_img="test/Output_data_batch{:0=3}".format(batch_idx),
                writer=self.writer
                )
