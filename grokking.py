"""
=============================================================================
EcoSort — Grokking Graph Generator
=============================================================================
Generates Welch Labs-style grokking visualizations from YOLO training results.

X-axis shows TRAINING STEPS (iterations), not epochs.
  steps = epoch × (dataset_size ÷ batch_size)

Usage:
  python grokking.py --csv results.csv --dataset-size 10000 --batch 16
  python grokking.py --csv r_10k.csv r_60k.csv --dataset-size 10000 60000 --batch 16
  python grokking.py --excel results.xlsx --dataset-size 60000 --batch 16
  python grokking.py --csv results.csv --dataset-size 10000 --batch 16 --detailed
  python grokking.py --csv results.csv --dataset-size 10000 --batch 16 --metric mAP50-95

Metrics: mAP50 (default), mAP50-95, precision, recall, loss
=============================================================================
"""

import argparse
import sys
import os
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
except ImportError:
    print("ERROR: pip install matplotlib")
    sys.exit(1)

try:
    import openpyxl  # noqa: F401
except ImportError:
    openpyxl = None


# ═══════════════════════════════════════════════════════════════════════
# WELCH LABS COLOR SCHEME
# ═══════════════════════════════════════════════════════════════════════
BG_COLOR = '#0a0a0a'
CYAN = '#65c8d0'
YELLOW = '#ffd35a'
GRID_COLOR = '#1a1a1a'
TEXT_COLOR = '#dfd0b9'
AXIS_COLOR = '#948979'


def load_results(filepath):
    """Load results from CSV or Excel file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ('.xlsx', '.xls'):
        if openpyxl is None:
            print("ERROR: pip install openpyxl (needed for Excel)")
            sys.exit(1)
        df = pd.read_excel(filepath)
    else:
        print(f"ERROR: Unsupported format '{ext}'. Use .csv or .xlsx")
        sys.exit(1)
    df.columns = df.columns.str.strip()
    return df


def find_column(df, candidates):
    """Find first matching column name from candidates list."""
    for col in candidates:
        for df_col in df.columns:
            if col.lower() == df_col.lower().strip():
                return df_col
    return None


def epochs_to_steps(epochs, dataset_size, batch_size):
    """Convert epoch numbers to cumulative training step counts."""
    iters_per_epoch = dataset_size // batch_size
    return epochs * iters_per_epoch


def extract_metrics(df, metric='mAP50'):
    """
    Extract train and val metrics from the DataFrame.

    Returns: (epochs, train_accuracy, val_accuracy)
    """
    epoch_col = find_column(df, ['epoch'])
    if epoch_col is None:
        print("ERROR: No 'epoch' column found.")
        sys.exit(1)
    epochs = df[epoch_col].values

    if metric == 'loss':
        train_box = find_column(df, ['train/box_loss'])
        train_cls = find_column(df, ['train/cls_loss'])
        train_dfl = find_column(df, ['train/dfl_loss'])
        val_box = find_column(df, ['val/box_loss'])
        val_cls = find_column(df, ['val/cls_loss'])
        val_dfl = find_column(df, ['val/dfl_loss'])

        train_loss = df[train_box].values + df[train_cls].values
        val_loss = df[val_box].values + df[val_cls].values
        if train_dfl:
            train_loss += df[train_dfl].values
        if val_dfl:
            val_loss += df[val_dfl].values

        max_loss = max(train_loss.max(), val_loss.max())
        train_acc = 1.0 - (train_loss / max_loss)
        val_acc = 1.0 - (val_loss / max_loss)
        return epochs, train_acc, val_acc

    # mAP50, mAP50-95, precision, recall
    metric_map = {
        'mAP50': ['metrics/mAP50(B)', 'mAP50(B)', 'mAP50', 'mAP_0.5'],
        'mAP50-95': ['metrics/mAP50-95(B)', 'mAP50-95(B)', 'mAP50-95'],
        'precision': ['metrics/precision(B)', 'precision(B)', 'precision'],
        'recall': ['metrics/recall(B)', 'recall(B)', 'recall'],
    }

    val_col = find_column(df, metric_map[metric])
    if val_col is None:
        print(f"ERROR: Column '{metric}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    val_acc = df[val_col].values

    # Training "accuracy" from inverted/normalized loss
    train_box = find_column(df, ['train/box_loss'])
    train_cls = find_column(df, ['train/cls_loss'])

    if train_box and train_cls:
        train_loss = df[train_box].values + df[train_cls].values
        train_dfl = find_column(df, ['train/dfl_loss'])
        if train_dfl:
            train_loss += df[train_dfl].values
        # Normalize [max_loss, min_loss] → [0, 1]
        train_acc = 1.0 - ((train_loss - train_loss.min()) /
                           (train_loss.max() - train_loss.min() + 1e-8))
        # Scale to match val range
        train_acc = train_acc * max(val_acc.max(), 0.95)
    else:
        train_acc = val_acc.copy()

    return epochs, train_acc, val_acc


def style_axis(ax):
    """Apply Welch Labs dark styling to an axis."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)


def plot_grokking(results_list, dataset_sizes, batch_size,
                  labels=None, metric='mAP50',
                  save_path=None, title=None, show=True):
    """Generate a Welch Labs-style grokking plot with TRAINING STEPS x-axis."""

    n = len(results_list)
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7), facecolor=BG_COLOR)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), facecolor=BG_COLOR)

    for idx, (filepath, df) in enumerate(results_list):
        ax = axes[idx]
        style_axis(ax)

        epochs, train_acc, val_acc = extract_metrics(df, metric)
        ds = dataset_sizes[idx] if idx < len(dataset_sizes) else dataset_sizes[0]
        steps = epochs_to_steps(epochs, ds, batch_size)

        # Thick lines matching Welch Labs
        ax.plot(steps, train_acc, color=CYAN, linewidth=2.8,
                label='TRAINING', alpha=0.95)
        ax.plot(steps, val_acc, color=YELLOW, linewidth=2.8,
                label='TESTING', alpha=0.95)

        ax.set_xlabel('TRAINING STEPS', fontsize=13,
                      color=TEXT_COLOR, fontweight='bold', labelpad=12)
        ax.set_ylabel('ACCURACY', fontsize=13,
                      color=TEXT_COLOR, fontweight='bold', labelpad=12)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(steps[0], steps[-1])

        # Legend — bottom right, boxed, colored labels
        legend = ax.legend(loc='lower right', fontsize=12,
                          facecolor='#1a1a1a', edgecolor=AXIS_COLOR,
                          labelcolor=[CYAN, YELLOW], framealpha=0.9)
        legend.get_frame().set_linewidth(0.8)

        if labels and idx < len(labels):
            ax.set_title(labels[idx], fontsize=14, color=TEXT_COLOR,
                        fontweight='bold', pad=15)

    if title is None:
        title = "Grokking: Models Learning After Long\nPeriods of Time"
    fig.suptitle(title, fontsize=20, color=TEXT_COLOR,
                fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        fig.savefig(save_path, dpi=300, facecolor=BG_COLOR,
                    edgecolor='none', bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def plot_grokking_detailed(filepath, df, dataset_size, batch_size,
                           metric='mAP50', save_path=None, show=True):
    """4-panel grokking analysis: curve, gap, losses, LR — all in steps."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG_COLOR)
    fig.suptitle(f"Grokking Analysis: {os.path.basename(filepath)}",
                fontsize=16, color=TEXT_COLOR, fontweight='bold', y=0.98)

    epochs, train_acc, val_acc = extract_metrics(df, metric)
    steps = epochs_to_steps(epochs, dataset_size, batch_size)

    # ── Panel 1: Grokking curve ──
    ax = axes[0, 0]
    style_axis(ax)
    ax.plot(steps, train_acc, color=CYAN, linewidth=2.5, label='TRAINING')
    ax.plot(steps, val_acc, color=YELLOW, linewidth=2.5, label='TESTING')
    ax.set_title('Train vs Test Accuracy', fontsize=12, color=TEXT_COLOR)
    ax.set_xlabel('TRAINING STEPS', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('ACCURACY', color=TEXT_COLOR, fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower right', facecolor='#1a1a1a', edgecolor=AXIS_COLOR,
              labelcolor=[CYAN, YELLOW])

    # ── Panel 2: Grokking gap ──
    ax = axes[0, 1]
    style_axis(ax)
    gap = train_acc - val_acc
    ax.fill_between(steps, gap, 0, color=CYAN, alpha=0.3)
    ax.plot(steps, gap, color=CYAN, linewidth=2)
    ax.axhline(y=0, color=AXIS_COLOR, linewidth=0.5, linestyle='--')
    ax.set_title('Grokking Gap (Train - Test)', fontsize=12, color=TEXT_COLOR)
    ax.set_xlabel('TRAINING STEPS', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('GAP', color=TEXT_COLOR, fontsize=10)

    # ── Panel 3: Training losses ──
    ax = axes[1, 0]
    style_axis(ax)
    loss_map = {
        'Box Loss': ('train/box_loss', '#e74c3c'),
        'Cls Loss': ('train/cls_loss', '#3498db'),
        'DFL Loss': ('train/dfl_loss', '#2ecc71'),
    }
    label_colors = []
    for name, (col_name, color) in loss_map.items():
        col = find_column(df, [col_name])
        if col:
            ax.plot(steps, df[col].values, color=color,
                    linewidth=1.8, label=name, alpha=0.85)
            label_colors.append(color)
    ax.set_title('Training Losses', fontsize=12, color=TEXT_COLOR)
    ax.set_xlabel('TRAINING STEPS', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('LOSS', color=TEXT_COLOR, fontsize=10)
    if label_colors:
        ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor=AXIS_COLOR,
                  labelcolor=label_colors)

    # ── Panel 4: Learning rate ──
    ax = axes[1, 1]
    style_axis(ax)
    lr_col = find_column(df, ['lr/pg0', 'lr/pg1', 'lr/pg2', 'lr'])
    if lr_col:
        ax.plot(steps, df[lr_col].values, color='#e67e22', linewidth=2)
    ax.set_title('Learning Rate Schedule', fontsize=12, color=TEXT_COLOR)
    ax.set_xlabel('TRAINING STEPS', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('LEARNING RATE', color=TEXT_COLOR, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        base, ext = os.path.splitext(save_path)
        detailed_path = f"{base}_detailed{ext}"
        fig.savefig(detailed_path, dpi=300, facecolor=BG_COLOR,
                    edgecolor='none', bbox_inches='tight')
        print(f"Saved: {detailed_path}")
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="EcoSort Grokking Graph Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python grokking.py --csv results.csv --dataset-size 10000 --batch 16
  python grokking.py --csv r_10k.csv r_60k.csv --dataset-size 10000 60000 --batch 16
  python grokking.py --csv results.csv --dataset-size 60000 --batch 16 --detailed
  python grokking.py --excel results.xlsx --dataset-size 10000 --batch 16 --metric mAP50-95
        """)

    parser.add_argument('--csv', nargs='+', help='Results CSV file(s)')
    parser.add_argument('--excel', nargs='+', help='Results Excel file(s)')
    parser.add_argument('--dataset-size', nargs='+', type=int, required=True,
                        help='Number of training images (e.g., 10000 or 60000). '
                             'Provide one per file or one for all.')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size used during training (default: 16)')
    parser.add_argument('--labels', nargs='+', help='Labels for each result set')
    parser.add_argument('--metric', default='mAP50',
                        choices=['mAP50', 'mAP50-95', 'precision', 'recall', 'loss'],
                        help='Metric to plot (default: mAP50)')
    parser.add_argument('--save', default=None, help='Save plot to file')
    parser.add_argument('--title', default=None, help='Custom plot title')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate 4-panel detailed analysis')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot')

    args = parser.parse_args()

    # Collect input files
    files = []
    if args.csv:
        files.extend(args.csv)
    if args.excel:
        files.extend(args.excel)
    if not files:
        default_paths = [
            'results.csv',
            'runs/ecosort/train_10k/results.csv',
            'runs/ecosort/train_60k/results.csv',
        ]
        for p in default_paths:
            if os.path.exists(p):
                files.append(p)
        if not files:
            print("ERROR: No input files. Use --csv or --excel")
            sys.exit(1)

    # Expand dataset_sizes to match number of files
    ds_sizes = args.dataset_size
    if len(ds_sizes) == 1:
        ds_sizes = ds_sizes * len(files)
    elif len(ds_sizes) != len(files):
        print(f"ERROR: Got {len(ds_sizes)} dataset sizes for {len(files)} files.")
        print("       Provide one size per file, or one size for all.")
        sys.exit(1)

    # Print step calculation
    for f, ds in zip(files, ds_sizes):
        iters = ds // args.batch
        print(f"  {os.path.basename(f)}: {ds} images / batch {args.batch} = {iters} iters/epoch")

    # Load results
    results_list = []
    for f in files:
        if not os.path.exists(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)
        df = load_results(f)
        results_list.append((f, df))
        print(f"  Loaded: {f} ({len(df)} epochs, {len(df) * (ds_sizes[0] // args.batch)} total steps)")

    labels = args.labels
    if labels is None and len(results_list) > 1:
        labels = [os.path.basename(f) for f, _ in results_list]

    save_path = args.save if args.save else 'grokking_plot.png'
    show = not args.no_show

    # Main grokking plot
    plot_grokking(results_list, ds_sizes, args.batch,
                  labels=labels, metric=args.metric,
                  save_path=save_path, title=args.title, show=show)

    # Detailed 4-panel
    if args.detailed:
        for i, (filepath, df) in enumerate(results_list):
            plot_grokking_detailed(filepath, df, ds_sizes[i], args.batch,
                                   metric=args.metric,
                                   save_path=save_path, show=show)

    print("\nDone.")


if __name__ == "__main__":
    main()
