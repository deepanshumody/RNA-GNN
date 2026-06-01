"""Imbalance-aware metrics for candidate-site ranking.

Locating ion sites is an extreme class-imbalance problem (positives are a tiny
fraction of all candidate grid points), so ROC AUC alone is misleading. This
reports ROC AUC together with PR-AUC, precision/recall at the Youden-J
threshold, and an *enrichment* number — what fraction of the score-ranked
candidates you must keep to recover the true site(s) — which reflects how the
model is actually used: as a first-pass filter that shrinks the search space.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binding_site_metrics(y_true, y_score):
    """Compute ranking / classification metrics for one set of candidates.

    Args:
        y_true: array-like of 0/1 labels.
        y_score: array-like of model scores (higher = more likely positive).

    Returns:
        ``dict`` of plain ``float`` / ``int`` values. The ROC/PR/threshold
        fields are ``nan`` when only one class is present (they are undefined).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n_total = int(y_true.size)
    n_pos = int(y_true.sum())

    out = {
        "n_total": n_total,
        "n_pos": n_pos,
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "youden_threshold": float("nan"),
        "precision_at_youden": float("nan"),
        "recall_at_youden": float("nan"),
        "n_pred_pos_at_youden": 0,
        "enrichment_top_fraction": float("nan"),
        "first_hit_fraction": float("nan"),
    }
    # ROC/PR and a threshold are only defined when both classes are present.
    if n_pos == 0 or n_pos == n_total:
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["pr_auc"] = float(average_precision_score(y_true, y_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    thr = float(thresholds[int(np.argmax(youden))])
    pred = y_score >= thr
    tp = int(np.sum(pred & (y_true == 1)))
    n_pred_pos = int(np.sum(pred))
    out["youden_threshold"] = thr
    out["n_pred_pos_at_youden"] = n_pred_pos
    out["precision_at_youden"] = float(tp / n_pred_pos) if n_pred_pos else 0.0
    out["recall_at_youden"] = float(tp / n_pos)

    # Enrichment: rank candidates by score (desc) and see how deep the positives
    # sit. Ties are broken pessimistically (positives ranked *after* negatives
    # of equal score) so the number is never optimistic.
    order = np.lexsort((y_true, -y_score))
    ranked_true = y_true[order]
    pos_ranks = np.flatnonzero(ranked_true == 1)  # 0-based ranks of positives
    out["first_hit_fraction"] = float((pos_ranks[0] + 1) / n_total)
    out["enrichment_top_fraction"] = float((pos_ranks[-1] + 1) / n_total)
    return out


def format_metrics(m):
    """One-line human-readable summary of :func:`binding_site_metrics` output."""
    return (
        "ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}  "
        "recall@J={recall_at_youden:.2f} precision@J={precision_at_youden:.3f} "
        "(pred+={n_pred_pos_at_youden}/{n_total})  "
        "enrichment: all {n_pos} site(s) within top {enrichment_top_fraction:.1%}"
    ).format(**m)
