import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoard_Record():
    def __init__(self, tsb_folder, tsb_dirname):
        self.tsb_path = os.path.join(tsb_folder, tsb_dirname)
        self.writer = SummaryWriter(self.tsb_path)
        
    def add_scalar(self, name_tag, y_value, x_value):
        self.writer.add_scalar(name_tag, y_value, x_value)

#tensorboard --logdir tensorboard --bind_all