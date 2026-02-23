import argparse
from .core import plot_dife, run_cl_benchmark, dife_ode, fit_dife, advance_simulation

def main():
    parser = argparse.ArgumentParser(description="DIFE CLI for Grok continual learning")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plot", help="Plot DIFE decay")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--Q0", type=float, default=1.0)
    p.add_argument("--n_max", type=int, default=50)

    b = sub.add_parser("bench", help="Run CL benchmark")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--beta", type=float, default=0.01)

    o = sub.add_parser("ode", help="ODE analysis")
    o.add_argument("--n_max", type=int, default=100)

    f = sub.add_parser("fit", help="Fit to CSV")
    f.add_argument("--csv", required=True)

    a = sub.add_parser("advance", help="Grok simulation")
    a.add_argument("--tasks", type=int, default=3)
    a.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    if args.command == "plot":
        plot_dife(args.alpha, args.beta, args.Q0, args.n_max)
    elif args.command == "bench":
        run_cl_benchmark(args.seeds, args.beta)
    elif args.command == "ode":
        dife_ode(args.n_max)
    elif args.command == "fit":
        fit_dife(args.csv)
    elif args.command == "advance":
        advance_simulation(args.tasks, args.epochs)

if __name__ == "__main__":
    main()
