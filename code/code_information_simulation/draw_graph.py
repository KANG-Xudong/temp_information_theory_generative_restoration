import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

root_dir = 'exp'
folder_name = 'Rain800_{}_lsgan_1'

model_list = ['unet5', 'unet5_residual', 'unet5_contextdilated', 'unet5_scan', 'unet5_recursive', 'unet5_attention', 'unet5_repeat', 'unet5_dense', 'unet5_ddb15x1', 'unet5_ddb15x1_scu', 'unet5_scu']
name_list = ['baseline', 'Residual', 'ContextualDilated', 'SCAN', 'Recursive', 'Attentive', 'Repeat', 'Dense', 'DDB', 'DDB + SCU', 'SCU']


#########################################################################################################################################################################################################

COLOR = ['#000000', '#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#02D0FF', '#0000FF', '#7000F0', '#FF00FF', '#F5A5A0', '#DD9575']

def smooth(data, weight=0.8):
    smoothed = []
    last = data[0]
    for d in data:
        smoothed_val = last * weight + (1 - weight) * d
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def read_data(csv_file):
    df = pd.read_csv(csv_file)
    return {'PSNR': df['PSNR'].mean(), 'SSIM': df['SSIM'].mean()}


def draw(data, title, set_name, xlabel, ylabel, ylim, xticklabels):
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.7, 1]}, figsize = (9, 5))
    i = 0
    for m in data:
        x, y = data[m]
        # plot line chart
        y_smooth = smooth(y)
        axs[0].plot(x, y, color=COLOR[i], alpha=0.2)
        axs[0].plot(x, y_smooth, color=COLOR[i], label=name_list[i])

        # plot scatter points
        axs[1].scatter(i, y_smooth[-1], color=COLOR[i])

        # add connection
        axs[1].add_artist(ConnectionPatch(xyA=(x[-1], y_smooth[-1]), xyB=(i, y_smooth[-1]), coordsA='data', coordsB='data', axesA=axs[0], axesB=axs[1], linestyle='dashed', color='dimgrey'))

        i += 1

    axs[0].legend(loc='lower right', prop={'size': 11.5})
    axs[1].set_xticks(np.arange(len(xticklabels)))
    if max([len(s) for s in xticklabels]) < 5:
        axs[1].set_xticklabels(xticklabels)
    else:
        axs[1].set_xticklabels(xticklabels, rotation=45, ha='right')

    axs[0].set_ylim(ylim[0], ylim[1])
    axs[1].set_ylim(ylim[0], ylim[1])

    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel('epoch')
    axs[1].set_ylabel(ylabel)
    if xlabel is not None:
        axs[1].set_xlabel(xlabel)

    fig.suptitle('%s - %s - %s' % (title, ylabel, set_name))
    plt.tight_layout(h_pad=0, w_pad=2, rect=[0, 0, 1, 0.93])
    plt.show()
    plt.clf()


def main(title, xlabel):
    for set_name in ['training_set', 'testing_set']:
        for ylabel, ylim in [('PSNR', [20, 28]), ('SSIM', [0.5, 1.0])]:
            csv_file_name = '{}_evaluation.csv'.format(set_name)
            data = {}
            for m in model_list:
                x, y = [], []
                for epoch in range(5, 205, 5):
                    csv_file = os.path.join(root_dir, folder_name.format(m), 'evaluate_log',
                                            'epoch_{}_1'.format(epoch), csv_file_name)
                    assert os.path.exists(csv_file), csv_file
                    x.append(epoch)
                    y.append(read_data(csv_file)[ylabel])
                data[m] = (x, y)

            draw(data, title, set_name, xlabel, ylabel, ylim, xticklabels=name_list)






#########################################################################################################################################################################################################

if __name__ == '__main__':

    title = 'Deraining Performances'

    xlabel = None

    main(title, xlabel)