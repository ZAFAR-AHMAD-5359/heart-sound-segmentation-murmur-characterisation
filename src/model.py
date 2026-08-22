"""
Classifier and training procedure.

Part of the reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

One classifier definition covers all four tasks. Only the number of output units
changes: two for murmur detection, three each for grade, timing and shape.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

SEED = 42


def fit_rf(X_train, y_train, seed=SEED):
    """The article's second gated classifier: 300 trees, balanced class
    weights. Deterministic given the seed."""
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                random_state=seed, n_jobs=-1)
    return rf.fit(X_train, y_train)


class LightweightMLP(nn.Module):
    """Two hidden layers with batch normalisation and dropout.

    Deliberately small. With roughly 200-300 murmur-positive recordings per
    characterisation task, a larger network memorises the training fold; the
    capacity here is matched to the label count, not to the feature count.
    """

    def __init__(self, n_features, n_classes, hidden=(128, 64), drops=(0.4, 0.3)):
        super().__init__()
        layers, prev = [], n_features
        for h, p in zip(hidden, drops):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    """Cross-entropy down-weighted on examples the model already classifies well.

    The characterisation labels are strongly imbalanced (for example grade I
    outnumbers grade III and above). Focal loss with class weighting stops the
    majority class from dominating the gradient.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight,
                                         reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def fit_mlp(X_train, y_train, groups_train, n_classes, device, seed=SEED,
            max_epochs=300, patience=5, eval_every=5):
    """Train one fold.

    Early stopping uses a validation split drawn INSIDE the training fold and
    disjoint by patient, so no participant informs both the stopping decision and
    the test score.
    """
    torch.manual_seed(seed)
    labels = list(range(n_classes))

    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed)
                  .split(X_train, y_train, groups=groups_train))

    Xt = torch.tensor(X_train[tr], dtype=torch.float32, device=device)
    yt = torch.tensor(y_train[tr], dtype=torch.long, device=device)
    Xv = torch.tensor(X_train[va], dtype=torch.float32, device=device)

    counts = np.bincount(y_train[tr], minlength=n_classes).astype(float)
    class_w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)

    model = LightweightMLP(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = FocalLoss(2.0, class_w)

    best, best_state, stale = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        criterion(model(Xt), yt).backward()
        opt.step()

        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                pred = model(Xv).argmax(1).cpu().numpy()
            score = f1_score(y_train[va], pred, average='macro',
                             labels=labels, zero_division=0)
            if score > best:
                best, stale = score, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model.eval()
