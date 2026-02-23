import argparse
from .core import Q_n, run_cl_benchmark

def main():
    parser = argparse.ArgumentParser(description="DIFE CLI for Grok continual learning")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("plot", help="Plot DIFE decay")
    bench = subparsers.add_parser("bench", help="Run Torch CL benchmark")
    args = parser.parse_args()
    if args.command == "bench":
        run_cl_benchmark()
        print("✅ Benchmark complete — see benchmark.csv & plots")
    else:
        n = np.arange(0, 51)
        Q_n(n)  # triggers plot in core if extended

if __name__ == "__main__":
    main()
