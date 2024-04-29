import json
import cv2
import numpy as np
import sys
import os
from torch.utils.data import Dataset
import torch
from share import *
from cldm.model import create_model
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/geometricShapes14k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/geometricShapes14k/' + source_filename)
        target = cv2.imread('./training/geometricShapes14k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


# Configs
#resume_path = './models/control_sd15_ini.ckpt'
resume_path ='./models/control_sd15_ini9_10__myds.pth'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 1



print("MODEL LOADING STARTING")
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
print("MODEL LOADING COMPLETE")


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=32, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator='gpu', devices=1, precision=32, callbacks=[logger], max_epochs=max_epochs, accumulate_grad_batches=4)

print("STARTING TRAINING")
torch.cuda.empty_cache()
# Train!
trainer.fit(model, dataloader)
torch.save(model.state_dict(), './models/control_sd15_ini10_10__myds.pth')
