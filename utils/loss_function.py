import torch


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, pred, y):
        return torch.sqrt(self.mse_loss(pred, y) + self.eps)
