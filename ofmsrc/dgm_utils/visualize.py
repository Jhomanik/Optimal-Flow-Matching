import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)


TICKS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16
global DEVICE
DEVICE='cpu'

PLT_CMAP = plt.cm.Spectral
PLT_NORM = plt.Normalize(vmin=0, vmax=1)


def plot_training_curves(train_losses, test_losses, logscale_y=False, logscale_x=False):
    n_train = len(train_losses[list(train_losses.keys())[0]])
    n_test = len(test_losses[list(train_losses.keys())[0]])
    x_train = np.linspace(0, n_test - 1, n_train)
    x_test = np.arange(n_test)

    plt.figure()
    for key, value in train_losses.items():
        plt.plot(x_train, value, label=key + '_train')

    for key, value in test_losses.items():
        plt.plot(x_test, value, label=key + '_test')

    if logscale_y:
        plt.semilogy()
    
    if logscale_x:
        plt.semilogx()

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Loss', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.grid()
    plt.show()


def show_samples(samples, title, figsize=None, nrow=None, save_file_path=None):
    if isinstance(samples, np.ndarray):
        samples = torch.FloatTensor(samples)
    if nrow is None:
        nrow = int(np.sqrt(len(samples)))
    grid_samples = make_grid(samples, nrow=nrow)

    grid_img = grid_samples.permute(1, 2, 0)
    if figsize is None:
        figsize = (6, 6)
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.imshow(grid_img)
    plt.axis('off')
    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path, dpi=100)


def visualize_images(data, title):
    idxs = np.random.choice(len(data), replace=False, size=(100,))
    images = data[idxs]
    show_samples(images, title)


def visualize_2d_data(train_data, test_data, train_labels=None, test_labels=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title('train', fontsize=TITLE_FONT_SIZE)
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
    ax1.tick_params(labelsize=LABEL_FONT_SIZE)
    ax2.set_title('test', fontsize=TITLE_FONT_SIZE)
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=1, c=test_labels)
    ax2.tick_params(labelsize=LABEL_FONT_SIZE)
    plt.show()


def visualize_2d_samples(data, title, labels=None, xlabel=None, ylabel=None, s=1):
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], s=s, c=labels)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.show()

def visualize_2d_samples_groups(
    samples_groups, 
    title, 
    colors=None, 
    labels=None, 
    xlabel=None, 
    ylabel=None, 
    s=1, 
    figsize=(5, 5),
    save_path=None,
    script_mode=False
):
    plt.figure(figsize=figsize)
    if colors is None:
        colors = np.linspace(0., 1., len(samples_groups))
    labels_is_none = False
    if labels is None:
        labels_is_none = True
        labels = [None,] * len(samples_groups)
    for samples, col, lbl in zip(samples_groups, colors, labels):
        np_samples = make_numpy(samples)
        if isinstance(col, float):
            col = PLT_CMAP(PLT_NORM(col))
        plt.scatter(
            np_samples[:, 0],
            np_samples[:, 1], 
            s=s, 
            c=[col,]*len(np_samples),
            label=lbl)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    if not labels_is_none:
        plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if not script_mode:
        plt.show()


def visualize_2d_densities(x_grid, y_grid, densities, title, xlabel=None, ylabel=None):
    densities = densities.reshape([y_grid.shape[0], y_grid.shape[1]])
    plt.figure(figsize=(5, 5))
    plt.pcolor(x_grid, y_grid, densities)
    plt.pcolor(x_grid, y_grid, densities)

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.show()
    
def visualize_2d_map(
    data, mapped_data, title,
    data_color='red', mapped_data_color='blue', map_color='green',
    data_label=None, mapped_data_label=None, map_label=None,
    xlabel=None, ylabel=None,
    s=1, linewidth=0.2, map_alpha=0.6, data_alpha=0.4,
    figsize=(5, 5), dpi=None, save_path=None,
    script_mode=False):

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(mapped_data, torch.Tensor):
        mapped_data = mapped_data.detach().cpu().numpy()
    lines = np.concatenate([data, mapped_data], axis=-1).reshape(-1, 2, 2)
    lc = matplotlib.collections.LineCollection(
        lines, color=map_color, linewidths=linewidth, alpha=map_alpha, label=map_label)
    ax.add_collection(lc)
    ax.scatter(
        data[:, 0], data[:, 1], s=s, label=data_label,
        alpha=data_alpha, zorder=2, color=data_color)
    ax.scatter(
        mapped_data[:, 0], mapped_data[:, 1], s=s, label=mapped_data_label,
        alpha=data_alpha, zorder=2, color=mapped_data_color)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    if save_path is not None:
        plt.savefig(save_path)
    if not script_mode:
        plt.show()
    plt.close()

def visualize_2d_torch_scalar_func(
    func, title, func_transform=lambda x: x,
    val=2., x_lim=None, y_lim=None, dx=0.025, dy=0.025, y_val=0.025,
    figsize=(12, 10), dpi=100, levels=200,
    device=None,
    xlabel=None, ylabel=None):

    if x_lim is None:
        x_lim = (-val, val)
    assert len(x_lim) == 2
    if y_lim is None:
        y_lim = (-val, val)
    assert len(y_lim) == 2

    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    if device is None:
        device = DEVICE
    mesh_xs = torch.tensor(np.stack([x, y], axis=2).reshape(-1, 2)).to(device).float()
    vals = func_transform(func(mesh_xs)).detach().cpu().numpy()
    vals = vals.reshape(x.shape)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    cf = ax.contourf(x, y, vals, levels)
    fig.colorbar(cf, ax=ax)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def visualize_2d_contour(
    density, X, Y, title, n_levels=3, 
    ax=None, xlabel=None, ylabel=None):

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot()
    density = density.reshape(X.shape)
    levels = np.linspace(np.min(density), np.max(density), n_levels)
    ax.contour(X, Y, density, levels=levels, c='red')
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    return ax

def visualize_2d_distribs(
    distribs, title, contour=True, density=True, 
    n_levels=20, x_lim=(-2., 2.), y_lim=(-2., 2.), dx=0.025, dy=0.025, 
    device=None,
    xlabel=None, ylabel=None):

    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    if device is None:
        device = DEVICE
    mesh_xs = torch.tensor(np.stack([x, y], axis=2).reshape(-1, 2)).to(device)
    densities = 0.
    if not isinstance(distribs, list):
        distribs = [distribs,]
    for distrib in distribs:
        densities += torch.exp(distrib.log_prob(mesh_xs)).detach().cpu().numpy()
    if contour:
        ax = visualize_2d_contour(
            densities, x, y, title='{} contour'.format(title), 
            n_levels=n_levels, xlabel=xlabel, ylabel=ylabel)
        plt.show()
    if density:
        visualize_2d_densities(
            x, y, densities, title='{} pdf'.format(title), 
            xlabel=xlabel, ylabel=ylabel)