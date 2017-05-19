import scipy as sp
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os


def save_result(name):
    result_dir = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, '%s.png' % name), dpi=300)
    plt.savefig(os.path.join(result_dir, '%s.svg' % name), dpi=300)
    return


def error_stats(x, y, w=None, verbose=False):
    if w is None:
        w = np.ones(len(y))
    mae = metrics.mean_absolute_error(x, y, w)
    mae_std = np.std(np.abs(x - y))
    rmse = np.sqrt(metrics.mean_squared_error(x, y, w))
    r2 = sp.stats.pearsonr(x, y)[0]
    R2 = metrics.r2_score(x, y)
    if verbose:
        print('MAE  = %.4f +/- %3.4f' % (mae, mae_std))
        print('RMSD = %.4f ' % (rmse))
        print('r^2  = %.3f , R^2  = %.3f ' % (r2, R2))

    return mae, mae_std, rmse, r2, R2


def data_scatter(data, label_x, label_y, c, title, cmap=None):
    x = data[label_x]
    y = data[label_y]
    if cmap is None:
        plt.scatter(x, y, c=c, s=50, alpha=0.75, label='')
    else:
        z = data[c]
        plt.scatter(x, y, c=z, s=50, alpha=0.75, label='', cmap=cmap)

    xmin, xmax = np.min(x), np.max(x)
    # ideal fit
    mae, mae_std, rmse, r2, R2 = error_stats(x, y, verbose=False)

    info_str = '\n'.join(['ideal fit',
                          '$R^2=%.3f$' % R2,
                          '$r^2=%.3f$' % r2,
                          'MAE =%2.3f (%3.2f)' % (mae, mae_std),
                          'RMSE  =%2.3f' % rmse])
    plt.plot([xmin, xmax], [xmin, xmax], ls='--',
             c='k', alpha=0.5, lw=3, label=info_str)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    sns.despine()
    plt.title(title)
    plt.legend(loc='upper left')
    if cmap is not None:
        plt.colorbar()
    plt.show()
    return


def basic_stats(y, verbose=False):
    mean_y, std_y = np.mean(y), np.std(y)
    min_y, max_y = np.min(y), np.max(y)
    low_y, high_y = mean_y - std_y * 2, mean_y + std_y * 2
    if verbose:
        print('Mean, std : %.3f +/- %.3f' % (mean_y, std_y))
        print('Min/Max   : [%.3f , %.3f] ' % (min_y, max_y))
        print('95%% range : [%.3f , %.3f] ' % (low_y, high_y))

    return mean_y, std_y, min_y, max_y, low_y, high_y


def linear_fit(x, y):
    alpha, beta, r1, p_value, std_err = sp.stats.linregress(x, y)
    polynomial = np.poly1d([alpha, beta])
    fit_y = polynomial(x)
    fit_x = (y - beta) / alpha
    mean, std, r1, r2 = error_stats(fit_y, y)

    return fit_x, fit_y, mean, std, r1, r2


def awesome_settings():
    # awesome plot options
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=2)
    sns.set_palette(sns.color_palette('bright'))
    # image stuff

    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.rcParams['savefig.dpi'] = 60
    plt.rcParams['lines.linewidth'] = 2
    # text stuff
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    return


def read_xyz(filename):

    if not os.path.exists(filename):
        raise ValueError("%s file not found" % filename)

    with open(filename) as xyz:
        n_atoms = int(xyz.readline().strip())
        title = xyz.readline().strip()
        iatom = 0
        atoms = []  # strings of maximum length 6
        coordinates = np.zeros((n_atoms, 3))
        for line in xyz:
            if not line.isspace():
                atom, x, y, z = line.split()
                try:
                    atoms.append(atom)
                    coordinates[iatom, :] = float(x), float(y), float(z)
                except IndexError:
                    raise ValueError(
                        "There are more coordinates to be read than indicated in the header.")
                iatom += 1

    if iatom != n_atoms:
        raise ValueError("number of coordinates read %d does not agree with number of atoms stated in file %d"
                         % (iatom, n_atoms))

    return atoms, coordinates


def xyztostr(atoms, coordinates):
    """write XYZ string

    """
    xyz_str = ''
    for atom, coord in zip(atoms, coordinates):
        xyz_str += '%s \t %2.6f %2.6f %2.6f \n' % (
            atom, coord[0], coord[1], coord[2])

    return xyz_str[:-2]


def color_properties(props):
    random_c = random.sample(sns.color_palette("Set2", 11), k=len(props))
    cmap_dict = {p: sns.light_palette(c, reverse=True, as_cmap=True,
                                      input="rgb") for p, c in zip(props, random_c)}
    col_dict = {p: c(0) for p, c in cmap_dict.items()}

    return col_dict, cmap_dict
