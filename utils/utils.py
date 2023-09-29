import os
import torch


def save_checkpoint(epoch, model, checkpoint_dir, best=False):
    if best:
        path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)
