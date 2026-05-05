import matplotlib.pyplot as plt

def plot_equity_curves(curves_dict, prefix="default"):
    plt.figure()

    for name, curve in curves_dict.items():
        plt.plot(curve, label=name)

    plt.legend()
    plt.title("Equity Curve Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Equity")

    plt.savefig(f"logs/{prefix}_equity.png")
    plt.close()