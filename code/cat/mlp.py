import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, n_inputs, n_features, dropout):
        super().__init__()
        self.dp = nn.Dropout(p=dropout)
        self.ac = nn.ReLU()
        self.fc_1 = nn.Linear(n_inputs, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.fc_4 = nn.Linear(64, 64)
        self.fc_5 = nn.Linear(64, n_features)

    def forward(self, x):
        for layer in (self.fc_1, self.fc_2, self.fc_3, self.fc_4):
            x = self.ac(self.dp(layer(x)))
        return self.fc_5(x)


class Predictor(nn.Module):
    def __init__(self, n_features, dropout):
        super().__init__()
        self.dp = nn.Dropout(p=dropout)
        self.ac = nn.ReLU()
        self.r_fc1 = nn.Linear(n_features, 64)
        self.r_fc2 = nn.Linear(64, 64)
        self.r_fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.ac(self.dp(self.r_fc1(x)))
        x = self.ac(self.dp(self.r_fc2(x)))
        return self.r_fc3(x)


class DomainHead(nn.Module):
    def __init__(self, n_features, dropout):
        super().__init__()
        self.d_fc1 = nn.Linear(n_features, 64)
        self.d_fc2 = nn.Linear(64, 64)
        self.d_fc3 = nn.Linear(64, 2)


def loader(arrays, batch_size, shuffle=True):
    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


class YieldMLP:
    def __init__(self, n_inputs, weighted, n_features=64, lr=1e-3, dropout=0.3, weight_decay=0.01):
        self.weighted = weighted
        self.encoder = Encoder(n_inputs, n_features, dropout)
        self.predictor = Predictor(n_features, dropout)
        self.domain_head = DomainHead(n_features, dropout)
        self.opt_e = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.opt_p = torch.optim.Adam(self.predictor.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched_e = StepLR(self.opt_e, 50, 0.5)
        self.sched_p = StepLR(self.opt_p, 50, 0.5)
        self.loss = nn.MSELoss(reduction="none") if weighted else nn.L1Loss()

    def source_rmse(self, monitor_loader):
        self.encoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            errors = [(self.predictor(self.encoder(batch[0])) - batch[1]) ** 2 for batch in monitor_loader]
        return float(torch.sqrt(torch.cat(errors).mean()))

    def fit(self, src_loader, tar_loader, monitor_loader, n_epochs=500, log_every=200):
        self.history = []
        for epoch in range(n_epochs):
            self.encoder.train()
            self.predictor.train()
            for src, tar in zip(src_loader, tar_loader):
                self.opt_e.zero_grad()
                self.opt_p.zero_grad()
                if self.weighted:
                    x, y, w = src
                    loss = torch.mean(self.loss(self.predictor(self.encoder(x)), y) * w)
                else:
                    x, y = src
                    features = self.encoder(x)
                    self.encoder(tar[0])
                    loss = self.loss(self.predictor(features), y)
                loss.backward()
                self.opt_p.step()
                self.opt_e.step()
            if epoch > 0 and epoch % log_every == 0:
                self.history.append((epoch, self.source_rmse(monitor_loader)))
            self.sched_e.step()
            self.sched_p.step()
        return self

    def predict(self, x):
        self.encoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            return self.predictor(self.encoder(torch.tensor(x, dtype=torch.float32))).numpy().ravel()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
