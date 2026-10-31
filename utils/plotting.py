"""Shared display style for the publication notebooks."""
from .datasets import flat_result_path, flat_result_directory
from pathlib import Path
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COLORS = {'coreset':'#C66F96','coreset_exp':'#B03A67','images':'#D6A15D',
          'labels':'#9184B3','random':'#67A7C1','random_exp':'#3477A1',
          'trajectory':'#BE824B','full':'#686868'}


def publication_style():
    """Use readable notebook sizes and editable text in PDF exports."""
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,
                         'figure.dpi':120,'savefig.dpi':180})


def muted_magma():
    """Match the muted image palette in the supplied Figure 2."""
    colors=mpl.colormaps['magma'](np.linspace(0,1,256))
    luminance=colors[:,:3]@np.array([0.2126,0.7152,0.0722])
    colors[:,:3]=0.30*colors[:,:3]+0.70*luminance[:,None]
    colors[:,:3]=0.94*colors[:,:3]+0.06*np.array([1.0,0.97,0.95])
    return mpl.colors.ListedColormap(colors,name='muted_magma')


def export_figure(figure, folder, name):
    """Save a vector PDF and a PNG preview while keeping the figure open."""
    folder=Path(folder);flat_result_directory(folder).mkdir(parents=True,exist_ok=True)
    for extension in ['pdf','png']:
        figure.savefig(flat_result_path(folder/f'{name}.{extension}'),bbox_inches='tight',facecolor='white')


def classification_table(simulation, experiment):
    """Show five simulation repetitions and ten measured repetitions."""
    rows = []
    assert np.array_equal(simulation['ipc_values'], experiment['size_values'])
    for column, size in enumerate(simulation['ipc_values']):
        row = {'Samples per class': int(size)}
        for label, key in [('Coreset', 'coreset'), ('Random', 'random')]:
            for kind, data in [('simulation', simulation), ('experiment', experiment)]:
                values = 100 * data[key + '_accuracy'][:, column]
                row[f'{label} {kind} (%)'] = f'{values.mean():.2f} ± {values.std(ddof=1):.2f}'
        rows.append(row)
    return pd.DataFrame(rows)


def plot_classification_sweep(simulation, experiment, title, ylim):
    """Plot five simulation repetitions and ten measured repetitions."""
    sizes = simulation['ipc_values']
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    handles = []
    for key, sim_color, exp_color, marker, line in [
        ('coreset', '#C75C89', '#B03A67', 'o', '-'),
        ('random', '#5EABC9', '#3477A1', 'D', '--'),
    ]:
        values = 100 * simulation[key + '_accuracy']
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        handle, = ax.plot(sizes, mean, color=sim_color, linestyle=line,
                          linewidth=2, label=('Coreset' if key == 'coreset' else 'Random') + ' (sim.)')
        ax.fill_between(sizes, mean - std, mean + std, color=sim_color, alpha=0.13, linewidth=0)
        values = 100 * experiment[key + '_accuracy']
        measured = ax.errorbar(sizes, values.mean(axis=0), yerr=values.std(axis=0, ddof=1),
                              color=exp_color, fmt=marker, linestyle='none', markersize=6,
                              markerfacecolor='white', markeredgewidth=1.4, capsize=3,
                              label=('Coreset' if key == 'coreset' else 'Random') + ' (exp.)')
        handles.extend([handle, measured])
    baseline = ax.axhline(100 * float(simulation['full_mean']), color='#686868', linestyle='--',
                         linewidth=1.5, label='Full training data')
    ax.set(xscale='log', ylim=ylim, title=title,
           xlabel='Training data size (samples/class)', ylabel='Test accuracy (%)')
    ax.set_xticks(sizes, labels=[str(int(v)) for v in sizes])
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[handles[0], handles[2], baseline, handles[1], handles[3]],
              frameon=False, fontsize=9, ncol=2, loc='lower right')
    return fig


def plot_classification_confusions(experiment, labels, training_size):
    """Use the paper's fixed support size and repetition for the measured pair."""
    count = len(labels)
    size = int(experiment['selected_confusion_size'])
    ratio = training_size / (size * count)
    title = f'{size} samples/class ({ratio:.0f}× training-data compression)'
    pink = LinearSegmentedColormap.from_list('classification_pink', ['#F8EAF0', '#E9B5CC', '#B84375'])
    blue = LinearSegmentedColormap.from_list('classification_blue', ['#EAF4F8', '#B7DDED', '#469DC9'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    for ax, key, cmap, caption, color in [
        (axes[0], 'coreset', pink, 'Synthetic coreset', '#B03A67'),
        (axes[1], 'random', blue, 'Random subset', '#3477A1'),
    ]:
        values = experiment[f'selected_{key}_confusion_percentages']
        ax.imshow(values, cmap=cmap, vmin=0, vmax=100, interpolation='nearest')
        ax.set_xticks(range(count), labels=labels, rotation=43, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(count), labels=labels)
        ax.tick_params(axis='both', width=0.6, length=2.5, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
        ax.set_title('Prediction', fontsize=11, pad=8)
        ax.set_ylabel('Ground truth', fontsize=11)
        ax.yaxis.set_label_position('right')
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            ax.text(column, row, f'{value:.0f}', ha='center', va='center', fontsize=10,
                    color='white' if value >= 55 else 'black')
        accuracy = 100 * float(experiment[f'selected_{key}_accuracy'])
        ax.text(0.5, -0.34, f'{caption}: {accuracy:.1f}% (experiment)', transform=ax.transAxes,
                ha='center', va='top', color=color, fontsize=12)
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.1, right=0.94, bottom=0.28, top=0.79, wspace=0.65)
    return fig
