"""Independent multi-day splits retaining the original baseline's test identities."""
from copy import deepcopy
from itertools import product

from wisig_protocol import ranked


def allocation(total, days):
    if not days or total < len(days):
        raise ValueError("Too few samples or no source days.")
    # Put the remainder on the last source day, declared before observing results.
    return {day: total // len(days) + (i >= len(days) - total % len(days)
            if total % len(days) else 0) for i, day in enumerate(days)}


def identities(groups):
    return {(g["relative_path"], i) for g in groups for i in g["signal_indices"]}


def build_split(rows, baseline, config, case):
    if config["target_date"] in case["dates"] or not set(case["dates"]) <= set(config["source_dates"]):
        raise ValueError("Training dates must be source-only.")
    if sorted(config["source_dates"])[-1] >= config["target_date"]:
        raise ValueError("Source dates must precede the target day.")
    roles, class_map = baseline["tx_roles"], baseline["class_map"]
    receivers = baseline["protocol"]["primary_rx_ids"]
    reserved = set().union(*(identities(baseline["views"]["test_" + day])
                            for day in baseline["protocol"]["test_dates"]))
    counts = allocation(config["train_per_tx_rx_total"], case["dates"])
    valid = allocation(config["validation_per_tx_rx_total"], case["dates"])
    common = allocation(config["validation_per_tx_rx_total"], config["source_dates"])
    views = {name: [] for name in ("train", "validation_known", "validation_unknown",
                                  "calibration_pool", "common_validation_known", "common_validation_unknown")}
    for row in rows:
        tx, day = row["tx_id"], row["capture_date"]
        if row["rx_id"] not in receivers or day not in config["source_dates"]:
            continue
        if tx not in roles["known"] + roles["validation_unknown"]:
            continue
        order = ranked(range(row["signal_count"]), baseline["protocol"]["split_seed"], row["relative_path"])
        order = [i for i in order if (row["relative_path"], i) not in reserved]
        known = tx in roles["known"]
        training = order[:120] if known else []
        validation = order[120:160] if known else order[:40]
        if len(validation) != 40 or (known and len(training) != 120):
            raise ValueError(f"Insufficient source samples: {tx}/{row['rx_id']}/{day}")
        def add(name, indices):
            views[name].append({**row, "tx_role": "known" if known else "validation_unknown",
                                "label": class_map.get(tx, -1), "signal_indices": sorted(indices)})
        if day in case["dates"]:
            if known:
                add("train", training[:counts[day]])
            add("validation_known" if known else "validation_unknown", validation[:valid[day]])
        if known:
            add("calibration_pool", training)
        add("common_validation_known" if known else "common_validation_unknown", validation[:common[day]])
    for day in baseline["protocol"]["test_dates"]:
        views["test_" + day] = deepcopy(baseline["views"]["test_" + day])
    plan = {"schema_version": "wisig_multiday_split_v1", "fold": baseline["fold"],
            "class_map": class_map, "tx_roles": roles, "provenance": baseline["provenance"],
            "case": case, "target_date": config["target_date"], "source_dates": config["source_dates"],
            "normalization": baseline["protocol"]["normalization"],
            "train_allocation": counts, "validation_allocation": valid,
            "common_validation_allocation": common, "views": views}
    validate_split(plan, rows, baseline)
    return plan


def validate_split(plan, rows, baseline):
    if plan["class_map"] != baseline["class_map"] or plan["tx_roles"] != baseline["tx_roles"]:
        raise ValueError("Transmitter roles differ from the baseline.")
    source = {row["relative_path"]: row for row in rows if row["relative_path"]}
    samples = {}
    receivers = baseline["protocol"]["primary_rx_ids"]
    for name, groups in plan["views"].items():
        keys = set()
        for group in groups:
            row = source[group["relative_path"]]
            if any(group.get(field) != value for field, value in row.items()):
                raise ValueError("Group metadata does not match the manifest.")
            indices = group["signal_indices"]
            if indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < row["signal_count"] for i in indices):
                raise ValueError("Invalid signal identities.")
            if group["label"] != plan["class_map"].get(group["tx_id"], -1):
                raise ValueError("Invalid transmitter label.")
            key = (group["tx_id"], group["rx_id"], group["capture_date"])
            if key in keys:
                raise ValueError("Duplicate groups.")
            keys.add(key)
            if not name.startswith("test_"):
                days = plan["case"]["dates"] if name in ("train", "validation_known", "validation_unknown") else plan["source_dates"]
                txs = plan["tx_roles"]["validation_unknown"] if name.endswith("unknown") else plan["tx_roles"]["known"]
                if group["tx_id"] not in txs or group["capture_date"] not in days:
                    raise ValueError("Unknown test transmitters or target date entered source data.")
                if name == "train":
                    count = plan["train_allocation"][group["capture_date"]]
                elif name.startswith("common_validation"):
                    count = plan["common_validation_allocation"][group["capture_date"]]
                elif name == "calibration_pool":
                    count = 120
                else:
                    count = plan["validation_allocation"][group["capture_date"]]
                if len(indices) != count:
                    raise ValueError("Source sample budget changed.")
        if name.startswith("test_"):
            if groups != baseline["views"][name]:
                raise ValueError("Test signals differ from the original baseline.")
        else:
            if keys != set(product(txs, receivers, days)):
                raise ValueError("Incomplete source coverage.")
        samples[name] = identities(groups)
    training, fit = samples["train"], samples["calibration_pool"]
    valid = samples["validation_known"] | samples["validation_unknown"]
    common = samples["common_validation_known"] | samples["common_validation_unknown"]
    test = set().union(*(values for name, values in samples.items() if name.startswith("test_")))
    if training & valid or (training | valid | fit | common) & test or fit & (valid | common) or training & common:
        raise ValueError("Training/calibration, validation and test overlap.")
    return {"verification_passed": True, "signal_counts": {key: len(value) for key, value in samples.items()},
            "identities_note": "NPZ file plus sample index; packet independence across Rx is not guaranteed."}
