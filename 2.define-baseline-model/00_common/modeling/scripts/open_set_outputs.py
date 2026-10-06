"""Export score-based predictions and common open-set diagnostics."""
import csv
from pathlib import Path

import numpy as np

from open_set_metrics import evaluate_open_set, operating_curve


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_scores(output, view, labels, predictions, scores, metadata, threshold, class_map,
                  *, arrays=None, per_signal=None):
    metrics = evaluate_open_set(labels, predictions, scores, threshold)
    inverse = {value: key for key, value in class_map.items()}
    accepted = np.where(scores >= threshold, predictions, -1)
    write_csv(output / f'{view}_predictions.csv', [{
        **row, 'true_label': int(labels[i]), 'closed_set_prediction': int(predictions[i]),
        'closed_set_predicted_tx': inverse[int(predictions[i])], 'score': float(scores[i]),
        'open_set_prediction': int(accepted[i]),
        'open_set_predicted_tx': inverse.get(int(accepted[i]), 'unknown'),
        **({key: value[i].item() for key, value in per_signal.items()} if per_signal else {}),
    } for i, row in enumerate(metadata)])
    np.savez_compressed(output / f'{view}_outputs.npz', labels=labels, predictions=predictions,
                        scores=scores, **(arrays or {}))
    curve = operating_curve(labels, predictions, scores)
    write_csv(output / f'{view}_oscr.csv', [
        {key: float(value[i]) for key, value in curve.items()}
        for i in range(len(curve['threshold']))])
    groups = []
    for field in ['tx_id', 'rx_id']:
        values = np.asarray([row[field] for row in metadata])
        for value in sorted(set(values)):
            mask = values == value
            groups.append({'view': view, 'group_by': field, 'group': value,
                           **evaluate_open_set(labels[mask], predictions[mask], scores[mask], threshold)})
    write_csv(output / f'{view}_subgroups.csv', groups)
    write_csv(output / f'{view}_confusion.csv', [{
        'true_label': true_label, 'predicted_label': predicted_label,
        'true_tx': inverse.get(true_label, 'unknown'), 'predicted_tx': inverse.get(predicted_label, 'unknown'),
        'count': int(((labels == true_label) & (accepted == predicted_label)).sum()),
    } for true_label in [-1, *range(len(class_map))] for predicted_label in [-1, *range(len(class_map))]])
    return metrics
