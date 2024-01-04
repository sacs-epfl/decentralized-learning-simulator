from time import sleep

import torch

from dasklearn.models import create_model

torch.multiprocessing.set_sharing_strategy('file_system')

print("Creating models in shared memory...")
models = []
for i in range(500):
    print("Creating model %d" % i)
    model = create_model("cifar10", architecture="resnet18")
    model = model.share_memory()
    models.append(model)

print("Sleeping...")
sleep(60)
