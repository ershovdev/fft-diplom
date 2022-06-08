"""Набор функций для построения графиков с результатами расчетов."""
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'normal',
    'font.size': 12,
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


def plot_dims(result, d_list, fpath=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    plt.subplots_adjust(wspace=0.3)

    ax1.set_xlabel('Размерность функции')
    ax1.set_title('Степень сжатия')

    ax2.set_xlabel('Размерность функции')
    ax2.set_title('Относительная ошибка')

    ax3.set_xlabel('Размерность функции')
    ax3.set_title('Время построения (сек.)')

    for name, data in result.items():
        c = [item['compression'] for item in data]
        e = [item['error'] for item in data]
        t = [item['time'] for item in data]

        ax1.plot(d_list, c, label=name,
            marker='s', markersize=8, linewidth=3)
        ax2.plot(d_list, e, label=name,
            marker='s', markersize=8, linewidth=3)
        ax3.plot(d_list, t, label=name,
            marker='s', markersize=8, linewidth=3)

    _prep_ax(ax1, xlog=False, ylog=True, leg=False)
    _prep_ax(ax2, xlog=False, ylog=True, leg=False)
    _prep_ax(ax3, xlog=False, ylog=True, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_dims_short(result, d_list, fpath=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)

    ax1.set_xlabel('Размерность функции')
    ax1.set_title('Степень сжатия')

    ax2.set_xlabel('Размерность функции')
    ax2.set_title('Время построения (сек.)')

    for name, data in result.items():
        c = [item['compression'] for item in data]
        t = [item['time'] for item in data]

        ax1.plot(d_list, c, label=name,
            marker='s', markersize=8, linewidth=3)
        ax2.plot(d_list, t, label=name,
            marker='s', markersize=8, linewidth=3)

    _prep_ax(ax1, xlog=False, ylog=True, leg=False)
    _prep_ax(ax2, xlog=False, ylog=True, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_dims_add(result, d_list, fpath=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    plt.subplots_adjust(wspace=0.3)

    ax1.set_xlabel('Размерность функции')
    ax1.set_title('Степень сжатия')

    ax2.set_xlabel('Размерность функции')
    ax2.set_title('Относительная ошибка')

    ax3.set_xlabel('Размерность функции')
    ax3.set_title('Степень ускорения')

    for name, data in result.items():
        c = [item['compression'] for item in data]
        e = [item['error'] for item in data]
        a = [item['acceleration'] for item in data]

        ax1.plot(d_list, c, label=name,
            marker='s', markersize=8, linewidth=3)
        ax2.plot(d_list, e, label=name,
            marker='s', markersize=8, linewidth=3)
        ax3.plot(d_list, a, label=name,
            marker='s', markersize=8, linewidth=3)

    _prep_ax(ax1, xlog=False, ylog=True, leg=False)
    _prep_ax(ax2, xlog=False, ylog=True, leg=False)
    _prep_ax(ax3, xlog=False, ylog=True, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_dims_add_short(result, d_list, fpath=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)

    ax1.set_xlabel('Размерность функции')
    ax1.set_title('Степень сжатия')

    ax2.set_xlabel('Размерность функции')
    ax2.set_title('Степень ускорения')

    for name, data in result.items():
        c = [item['compression'] for item in data]
        a = [item['acceleration'] for item in data]

        ax1.plot(d_list, c, label=name,
            marker='s', markersize=8, linewidth=3)
        ax2.plot(d_list, a, label=name,
            marker='s', markersize=8, linewidth=3)

    _prep_ax(ax1, xlog=False, ylog=True, leg=False)
    _prep_ax(ax2, xlog=False, ylog=True, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_dims_conv(result, d_list, fpath=None):
    return plot_dims_add_short(result, d_list, fpath)


def plot_dims_mul(result, d_list, fpath=None):
    return plot_dims_add(result, d_list, fpath)


def plot_dims_mul_short(result, d_list, fpath=None):
    return plot_dims_add_short(result, d_list, fpath)


def _prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()

    if leg:
        ax.legend(loc='best', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)
