"""Check the actual interpreter and shared baseline imports without changing them."""
import argparse
from importlib.metadata import version
import platform
import sys

from common import ROOT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(ROOT / "output/environment.json"))
    args = parser.parse_args()
    import numpy as np
    import torch
    from cnn_backbone import IQClassifier
    from openmax import OpenMax
    from open_set_metrics import evaluate_open_set
    torch.set_num_threads(2)
    model = IQClassifier(2, [32, 64, 128], [7, 5, 3], 128, .2).eval()
    with torch.inference_mode():
        result = model(torch.zeros(2, 2, 256))
    # Import alone does not exercise libMR's native fitting routine.
    logits = np.array([[5 + i / 100, 1] for i in range(40)] + [[1, 5 + i / 100] for i in range(40)])
    fitted = OpenMax(2, 20).fit(logits, np.repeat([0, 1], 40))
    assert np.isfinite(fitted.probabilities(logits, 1)).all()
    report = {"checked_at_jst": now(), "status": "passed", "python": sys.executable,
              "version": sys.version, "platform": platform.platform(), "device": "cpu",
              "packages": {key: version(key) for key in ["numpy", "torch", "scipy", "scikit-learn", "libmr", "Cython"]},
              "cnn_output_shape": list(result.shape), "native_libmr_fit": "passed"}
    write_json(args.output, report)
    print(f"Environment passed: {sys.executable}")


if __name__ == "__main__":
    main()
