"""Separate publication panels for output regression and autonomous forecasting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

CORE_SIM, CORE_EXP = '#C75C89', '#B03A67'
RAND_SIM, RAND_EXP, TRUTH = '#5EABC9', '#3477A1', '#666666'


def result_table(simulation, experiment):
    """Format repetition-level mean and sample standard deviation."""
    result = {'Training samples': simulation['support_sizes']}
    for source, suffix in [(simulation, 'sim.'), (experiment, 'exp.')]:
        for key, label in [('coreset', 'Coreset'), ('random', 'Random')]:
            values = source[f'{key}_r2']
            result[f'{label} ({suffix})'] = [f'{mean:.3f} ± {std:.3f}' for mean, std in
                                            zip(values.mean(axis=0), values.std(axis=0, ddof=1))]
    return pd.DataFrame(result).set_index('Training samples')


def _style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(direction='out', length=3, width=.8)


def regression_sweep(simulation, experiment, *, title, task):
    sizes = np.asarray(simulation['support_sizes'])
    figure, axis = plt.subplots(figsize=(6.2, 4.6), layout='constrained')
    for name, sim_color, exp_color, marker, line, label in [
        ('coreset', CORE_SIM, CORE_EXP, 'o', '-', 'Coreset'),
        ('random', RAND_SIM, RAND_EXP, 'D', '--', 'Random')]:
        values = simulation[f'{name}_r2'];mean = values.mean(axis=0);std = values.std(axis=0, ddof=1)
        axis.plot(sizes, mean, color=sim_color, ls=line, lw=1.8, label=f'{label} (sim.)')
        axis.fill_between(sizes, mean-std, mean+std, color=sim_color, alpha=.13, linewidth=0)
        values = experiment[f'{name}_r2']
        axis.errorbar(sizes, values.mean(axis=0), yerr=values.std(axis=0, ddof=1),
                      fmt=marker, color=exp_color, mfc='white', ms=6, capsize=3, lw=1,
                      label=f'{label} (exp.)')
    axis.axhline(float(simulation['full_r2']), color=TRUTH, ls='--', lw=1.2, label='Full training data')
    axis.set_xscale('log');axis.xaxis.set_major_locator(FixedLocator(sizes))
    axis.set_xticklabels([str(s) for s in sizes]);axis.xaxis.set_minor_formatter(NullFormatter())
    threshold, factor = (.4, .24) if task == 'ccpp' else (0., .18)
    def forward(x):return np.where(np.asarray(x)>=threshold, x, threshold+factor*(np.asarray(x)-threshold))
    def inverse(x):return np.where(np.asarray(x)>=threshold, x, threshold+(np.asarray(x)-threshold)/factor)
    axis.set_yscale('function', functions=(forward, inverse))
    axis.set_ylim((-.15, 1.02) if task == 'ccpp' else (-1.85, 1.06))
    axis.set_yticks([0,.4,.6,.8,1] if task == 'ccpp' else [-1.5,-.5,0,.25,.5,.75,1])
    axis.set(xlabel='Training data size (samples)', ylabel=r'Test $R^2$', title=title)
    _style(axis)
    axis.legend(loc='lower right' if task=='ccpp' else 'lower right', frameon=False, fontsize=8)
    figure.canvas.draw()
    y=axis.transAxes.inverted().transform(axis.transData.transform((sizes[0],threshold)))[1]
    for offset in [-.008,.008]:
        axis.plot([-.009,.009],[y+offset-.008,y+offset+.008],transform=axis.transAxes,color=TRUTH,clip_on=False,lw=.8)
    return figure


def regression_scatter(truth, prediction, *, method):
    color = CORE_EXP if method == 'Coreset' else RAND_EXP
    truth=np.asarray(truth);prediction=np.asarray(prediction)
    r2=1-np.sum((prediction-truth)**2)/np.sum((truth-truth.mean())**2)
    figure, axis=plt.subplots(figsize=(4.8,4.6),layout='constrained')
    axis.scatter(truth,prediction,s=10,color=color,alpha=.5,edgecolors='none',rasterized=True)
    axis.plot([424,496],[424,496],'--',color='#888888',lw=1)
    axis.set(xlim=(424,496),ylim=(424,496),xlabel='True output',ylabel='Predicted output',
             title=f'{method}: 2 training samples\n'+rf'$R^2={r2:.3f}$')
    axis.set_aspect('equal');axis.set_xticks([430,450,470,490]);axis.set_yticks([430,450,470,490])
    axis.tick_params(direction='out');return figure


def forecasting_trajectory(truth, prediction, *, method):
    color = CORE_EXP if method == 'Coreset' else RAND_EXP
    truth=np.asarray(truth);prediction=np.asarray(prediction);steps=np.arange(1,len(truth)+1)
    r2=1-np.sum((prediction-truth)**2)/np.sum((truth-truth.mean())**2)
    correlation=float(np.corrcoef(truth,prediction)[0,1])
    figure, axis=plt.subplots(figsize=(8,3.1),layout='constrained')
    axis.plot(steps,truth,color=TRUTH,lw=1.1,label='Ground truth')
    axis.plot(steps,prediction,color=color,lw=1.3,label=method)
    axis.set(xlim=(1,200),xlabel='Time step',ylabel='Value',
             title=f'{method}: 12 training samples\n'+rf'$R^2={r2:.3f},\ C={correlation:.3f}$')
    axis.set_xticks([1,50,100,150,200]);axis.legend(frameon=False,fontsize=9,ncol=2)
    _style(axis);return figure
