import torch
import torchmetrics
import numpy as np
from sklearn import metrics
from math import sqrt
from scipy import stats
from torchmetrics import Metric

class NPMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.add_state("preds", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds = torch.cat([self.preds, preds.detach().float().reshape(-1)])
        self.targets = torch.cat([self.targets, target.detach().float().reshape(-1)])

    def compute(self) -> torch.Tensor:
        y = self.targets.cpu().numpy()
        p = self.preds.cpu().numpy()
        return torch.tensor(float(self.func(y, p)), dtype=torch.float32)



def roc_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the ROC AUC score."""
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def pr_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the Precision-Recall AUC score."""
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Root Mean Squared Error."""
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Mean Squared Error."""
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient."""
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Spearman correlation coefficient."""
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Concordance Index."""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
    
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))



class CI(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.flatten()
        target = target.flatten()
        self.y_true.append(target)
        self.y_pred.append(preds)

    def compute(self):
        # Support both list-of-tensors (from append) and tensor states
        if isinstance(self.y_true, list):
            y_true_t = torch.cat([t.detach().float().flatten() for t in self.y_true], dim=0)
        else:
            y_true_t = self.y_true.detach().float().flatten()

        if isinstance(self.y_pred, list):
            y_pred_t = torch.cat([t.detach().float().flatten() for t in self.y_pred], dim=0)
        else:
            y_pred_t = self.y_pred.detach().float().flatten()

        y_true = y_true_t.cpu().numpy()
        y_pred = y_pred_t.cpu().numpy()
        return torch.tensor(ci(y_true, y_pred), dtype=torch.float32)


class RM2(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.flatten()
        target = target.flatten()
        self.y_true.append(target)
        self.y_pred.append(preds)

    def compute(self):
        # Support both list-of-tensors (from append) and tensor states
        if isinstance(self.y_true, list):
            y_true_t = torch.cat([t.detach().float().flatten() for t in self.y_true], dim=0)
        else:
            y_true_t = self.y_true.detach().float().flatten()

        if isinstance(self.y_pred, list):
            y_pred_t = torch.cat([t.detach().float().flatten() for t in self.y_pred], dim=0)
        else:
            y_pred_t = self.y_pred.detach().float().flatten()

        y_true = y_true_t.cpu().numpy()
        y_pred = y_pred_t.cpu().numpy()
        return torch.tensor(get_rm2(y_true, y_pred), dtype=torch.float32)
