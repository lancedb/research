"""Download pinned data and model snapshots before timed benchmark stages."""
import argparse
from huggingface_hub import snapshot_download
import bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=bench.DATASETS, default=bench.DATASETS)
    parser.add_argument("--models", action="store_true", help="Also download all open-model weights")
    args = parser.parse_args()
    for name in args.datasets:
        # Load prepares Arrow caches too. Corpus text remains disk-backed.
        bench.load(name)
    models = bench.REVISIONS["models"] if args.models else {bench.EMBEDDER: bench.revision(bench.EMBEDDER)}
    for model, revision in models.items():
        snapshot_download(model, revision=revision,
                          ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"])


if __name__ == "__main__":
    main()
