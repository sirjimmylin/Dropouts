"""Evaluate SeCo / TeCo / CACo pretrained encoders on EuroSAT.

Runs, for each model:
  1) Feature extraction on train + val (cached to .npz).
  2) Linear probe: frozen backbone + a Linear(512 -> 10) trained 100 epochs
     with Adam + MultiStepLR, matching main_eurosat.py.
  3) KNN (k=5, cosine) baseline on val using train features.
  4) Retrieval metrics on val: Precision@K and Recall@K (K in {1,5,10,20,100}),
     cosine similarity, each sample as a query against the remaining val set
     (self-match excluded).

All artifacts saved under CACo/eval_results/eurosat/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18

ROOT = Path('/oscar/home/vsmokash/deep-learning/CACo')
sys.path.insert(0, str(ROOT / 'src'))

from datasets.eurosat_dataset import EurosatDataset  # noqa: E402

DATA_DIR = ROOT / 'data' / 'EuroSAT_RGB'
OUT_DIR = ROOT / 'eval_results' / 'eurosat'
FEAT_DIR = OUT_DIR / 'features'
LP_DIR = OUT_DIR / 'linear_probe'
RET_DIR = OUT_DIR / 'retrieval'
LOG_DIR = OUT_DIR / 'logs'
for d in (FEAT_DIR, LP_DIR, RET_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODELS = {
    'random': 'random',
    'imagenet': 'imagenet',
    'seco': ROOT / 'checkpoints/resnet18-clean_10k_geography-seco-ep400-bs256-q4096/resnet18_seco_geo_10k_400.pth',
    'teco': ROOT / 'checkpoints/resnet18-clean_10k_geography-teco-ep400-bs256-q4096/resnet18_teco_geo_10k_400.pth',
    'caco': ROOT / 'checkpoints/resnet18-clean_10k_geography-caco-ep400-bs256-q4096/resnet18_caco_geo_10k_400.pth',
}
NUM_CLASSES = 10
IN_FEATURES = 512
BATCH_SIZE = 256
LP_EPOCHS = 100
SEED = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def build_backbone(spec):
    if isinstance(spec, str) and spec == 'random':
        torch.manual_seed(SEED)
        m = resnet18(pretrained=False)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten()).eval().to(device)
    if isinstance(spec, str) and spec == 'imagenet':
        m = resnet18(pretrained=True)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten()).eval().to(device)
    model = nn.Sequential(*list(resnet18().children())[:-1], nn.Flatten())
    state = torch.load(str(spec), map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval().to(device)


def get_loader(split, shuffle=False):
    ds = EurosatDataset(str(DATA_DIR), split=split, transform=T.ToTensor(), get_path=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True), ds


@torch.no_grad()
def extract_features(backbone, loader):
    feats, labels, paths = [], [], []
    for x, y, p in loader:
        f = backbone(x.to(device, non_blocking=True))
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        paths.extend(list(p))
    return np.concatenate(feats), np.concatenate(labels), np.array(paths)


def linear_probe(train_feats, train_labels, val_feats, val_labels, model_name):
    torch.manual_seed(SEED)
    classifier = nn.Linear(IN_FEATURES, NUM_CLASSES).to(device)
    optimizer = optim.Adam(classifier.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * LP_EPOCHS), int(0.8 * LP_EPOCHS)]
    )
    criterion = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(train_feats).to(device)
    Ytr = torch.from_numpy(train_labels).long().to(device)
    Xvl = torch.from_numpy(val_feats).to(device)
    Yvl = torch.from_numpy(val_labels).long().to(device)

    n_train = Xtr.size(0)
    history = []
    best_val_acc = 0.0
    best_val_state = None

    for epoch in range(LP_EPOCHS):
        classifier.train()
        perm = torch.randperm(n_train, device=device)
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            logits = classifier(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)
        scheduler.step()
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(Xvl)
            val_loss = criterion(val_logits, Yvl).item()
            val_pred = val_logits.argmax(1)
            val_acc = (val_pred == Yvl).float().mean().item()

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

        if epoch % 10 == 0 or epoch == LP_EPOCHS - 1:
            log(f'  [{model_name} LP] ep={epoch:3d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

    # Final epoch + best by val_acc
    final_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    classifier.eval()
    with torch.no_grad():
        final_logits = classifier(Xvl)
        final_pred = final_logits.argmax(1).cpu().numpy()

    # Best-epoch predictions
    classifier.load_state_dict(best_val_state)
    classifier.eval()
    with torch.no_grad():
        best_logits = classifier(Xvl)
        best_pred = best_logits.argmax(1).cpu().numpy()
        best_probs = F.softmax(best_logits, dim=1).cpu().numpy()

    return {
        'history': history,
        'final_state': final_state,
        'best_val_state': best_val_state,
        'final_val_acc': history[-1]['val_acc'],
        'best_val_acc': best_val_acc,
        'final_pred': final_pred,
        'best_pred': best_pred,
        'best_probs': best_probs,
    }


def knn_eval(train_feats, train_labels, val_feats, val_labels, k=5):
    tr = F.normalize(torch.from_numpy(train_feats).to(device), dim=1)
    vl = F.normalize(torch.from_numpy(val_feats).to(device), dim=1)
    sims = vl @ tr.T  # [n_val, n_train]
    topk = sims.topk(k, dim=1).indices  # [n_val, k]
    neighbor_labels = torch.from_numpy(train_labels).to(device)[topk]  # [n_val, k]
    # Majority vote
    pred = torch.mode(neighbor_labels, dim=1).values.cpu().numpy()
    acc = float((pred == val_labels).mean())
    return acc, pred


def per_class_report(pred, y_true, class_names):
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, pred):
        cm[int(t), int(p)] += 1
    out = {}
    for i, c in enumerate(class_names):
        support = int(cm[i].sum())
        tp = int(cm[i, i])
        recall = tp / support if support else 0.0
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'support': support,
        }
    overall_acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    macro_p = float(np.mean([v['precision'] for v in out.values()]))
    macro_r = float(np.mean([v['recall'] for v in out.values()]))
    macro_f1 = float(np.mean([v['f1'] for v in out.values()]))
    return {
        'confusion_matrix': cm.tolist(),
        'per_class': out,
        'accuracy': overall_acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
    }


def retrieval_metrics(val_feats, val_labels, ks=(1, 5, 10, 20, 100)):
    F_ = F.normalize(torch.from_numpy(val_feats).to(device), dim=1)
    sim = F_ @ F_.T
    sim.fill_diagonal_(-1e9)  # exclude self
    sort_idx = sim.argsort(dim=1, descending=True).cpu().numpy()  # [n, n-1 effectively]
    labels_t = val_labels
    # retrievals[i, j] = 1 if labels[sort_idx[i, j]] == labels[i]
    retrievals = (labels_t[sort_idx] == labels_t[:, None]).astype(np.float32)

    total_relevant_per_query = retrievals.sum(axis=1)  # number of same-class items available
    out = {}
    for k in ks:
        topk = retrievals[:, :k]
        precision_at_k = float(topk.mean() * 100)  # mean over queries and positions
        # Mean recall@k: for each query, fraction of its total relevant items recovered in top-k
        recall_per_q = np.where(
            total_relevant_per_query > 0,
            topk.sum(axis=1) / np.maximum(total_relevant_per_query, 1),
            0.0,
        )
        recall_at_k = float(recall_per_q.mean() * 100)
        # Also report the notebook-style recall (sum(topk) / sum(all relevants))
        notebook_recall = float(topk.sum() / retrievals.sum() * 100)
        out[f'P@{k}'] = precision_at_k
        out[f'R@{k}'] = recall_at_k
        out[f'R@{k}_notebook_style'] = notebook_recall
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f'Device: {device}')
    log(f'Data dir: {DATA_DIR}')

    # Build datasets once to read class_names
    _, _train_ds = get_loader('train', shuffle=False)
    _, _val_ds = get_loader('val', shuffle=False)
    class_names = _train_ds.classes
    log(f'Classes: {class_names}')
    log(f'Train size: {len(_train_ds)} | Val size: {len(_val_ds)}')

    summary = {
        'config': {
            'batch_size': BATCH_SIZE,
            'linear_probe_epochs': LP_EPOCHS,
            'seed': SEED,
            'num_classes': NUM_CLASSES,
            'class_names': class_names,
            'train_size': len(_train_ds),
            'val_size': len(_val_ds),
            'device': device,
        },
        'models': {},
    }

    for model_name, ckpt_path in MODELS.items():
        log(f'==================== {model_name.upper()} ====================')
        log(f'Checkpoint: {ckpt_path}')

        train_npz = FEAT_DIR / f'{model_name}_train.npz'
        val_npz = FEAT_DIR / f'{model_name}_val.npz'
        if train_npz.exists() and val_npz.exists():
            log(f'Loading cached features ({train_npz.name}, {val_npz.name})')
            tr_npz = np.load(train_npz, allow_pickle=True)
            vl_npz = np.load(val_npz, allow_pickle=True)
            f_tr, y_tr, p_tr = tr_npz['features'], tr_npz['labels'], tr_npz['paths']
            f_vl, y_vl, p_vl = vl_npz['features'], vl_npz['labels'], vl_npz['paths']
            log(f'  train feats: {f_tr.shape} | val feats: {f_vl.shape}')
        else:
            backbone = build_backbone(ckpt_path)
            train_loader, _ = get_loader('train', shuffle=False)
            val_loader, _ = get_loader('val', shuffle=False)
            log('Extracting train features...')
            f_tr, y_tr, p_tr = extract_features(backbone, train_loader)
            log(f'  train feats: {f_tr.shape}')
            log('Extracting val features...')
            f_vl, y_vl, p_vl = extract_features(backbone, val_loader)
            log(f'  val feats: {f_vl.shape}')
            np.savez(train_npz, features=f_tr, labels=y_tr, paths=p_tr)
            np.savez(val_npz, features=f_vl, labels=y_vl, paths=p_vl)
            del backbone
            torch.cuda.empty_cache()

        # KNN baseline
        knn_acc, knn_pred = knn_eval(f_tr, y_tr, f_vl, y_vl, k=5)
        log(f'KNN(k=5, cosine) val_acc: {knn_acc:.4f}')

        # Linear probe
        log('Training linear probe (100 epochs)...')
        lp = linear_probe(f_tr, y_tr, f_vl, y_vl, model_name)

        # Save probe state + history
        torch.save({
            'final_state': lp['final_state'],
            'best_val_state': lp['best_val_state'],
            'final_val_acc': lp['final_val_acc'],
            'best_val_acc': lp['best_val_acc'],
            'history': lp['history'],
            'in_features': IN_FEATURES,
            'num_classes': NUM_CLASSES,
            'class_names': class_names,
        }, str(LP_DIR / f'{model_name}_classifier.pt'))

        with open(LP_DIR / f'{model_name}_history.json', 'w') as f:
            json.dump(lp['history'], f, indent=2)

        np.savez(
            LP_DIR / f'{model_name}_predictions.npz',
            val_pred_final=lp['final_pred'],
            val_pred_best=lp['best_pred'],
            val_probs_best=lp['best_probs'],
            val_labels=y_vl,
            val_paths=p_vl,
            knn_pred=knn_pred,
        )

        # Metrics (linear probe, best epoch)
        lp_report = per_class_report(lp['best_pred'], y_vl, class_names)
        knn_report = per_class_report(knn_pred, y_vl, class_names)

        # Retrieval
        log('Computing retrieval metrics...')
        ret = retrieval_metrics(f_vl, y_vl)
        for k, v in ret.items():
            log(f'  {k}: {v:.2f}')

        summary['models'][model_name] = {
            'checkpoint': str(ckpt_path),
            'linear_probe': {
                'final_val_acc': lp['final_val_acc'],
                'best_val_acc': lp['best_val_acc'],
                'macro_precision': lp_report['macro_precision'],
                'macro_recall': lp_report['macro_recall'],
                'macro_f1': lp_report['macro_f1'],
                'per_class': lp_report['per_class'],
                'confusion_matrix': lp_report['confusion_matrix'],
            },
            'knn5_cosine': {
                'val_acc': knn_acc,
                'macro_f1': knn_report['macro_f1'],
                'per_class': knn_report['per_class'],
                'confusion_matrix': knn_report['confusion_matrix'],
            },
            'retrieval_val': ret,
        }

        log(f'[{model_name}] LP best val_acc={lp["best_val_acc"]:.4f} | final={lp["final_val_acc"]:.4f} | KNN={knn_acc:.4f}')

    with open(OUT_DIR / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'Wrote summary: {OUT_DIR / "metrics_summary.json"}')

    # Print a compact comparison table
    log('\n================= SUMMARY =================')
    log(f'{"model":<8} {"LP_best":>8} {"LP_final":>9} {"KNN5":>7} {"P@20":>7} {"R@100":>7} {"macroF1":>8}')
    for m, r in summary['models'].items():
        lp_ = r['linear_probe']
        log(f'{m:<8} {lp_["best_val_acc"]:>8.4f} {lp_["final_val_acc"]:>9.4f} '
            f'{r["knn5_cosine"]["val_acc"]:>7.4f} '
            f'{r["retrieval_val"]["P@20"]:>7.2f} {r["retrieval_val"]["R@100"]:>7.2f} '
            f'{lp_["macro_f1"]:>8.4f}')


if __name__ == '__main__':
    main()
