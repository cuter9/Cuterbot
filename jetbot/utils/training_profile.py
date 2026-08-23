# %matplotlib inline
# from IPython.display import clear_output
import time

import numpy as np
import os
import matplotlib

# matplotlib.use("TkAgg")
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# dir_training_records = os.path.join(dir_depo, 'training records', TRAIN_MODEL)
# os.makedirs(dir_training_records, exist_ok=True)
plt.close('all')
fig_1, ax_1 = plt.subplots(figsize=(14, 6))

font = {'fontweight': 'normal', 'fontsize': 16}
font_title = {'fontweight': 'normal', 'fontsize': 20}

# plot the training convergence profile
def plot_loss(loss_data, best_loss, best_epoch, no_epoch, epoch, info_train,
              dir_training_records, train_model, train_method, processor,
              show_training_plot=False, save_plot=False):
    # print(f"【除錯訊息】當前呼叫 plot_loss 的 save_plot 狀態為: {save_plot}")
    plt.cla()
    ax_1.tick_params(axis='both', labelsize='large')
    epochs = range(len(loss_data))
    ld_train = [ld[0] for ld in loss_data]
    ld_test = [ld[1] for ld in loss_data]
    ax_1.semilogy(epochs, ld_train, "r-", linewidth=2.0, label="Training Loss: {:.4E}".format(ld_train[-1]))
    ax_1.semilogy(epochs, ld_test, 'bs--', linewidth=2.0, label="Test Loss: {:.4E}".format(ld_test[-1]))

    # 避免 epochs 只有 0 或 1 時計算錯誤
    xlim = epochs[-1] + 2 if len(epochs) > 0 else 2
    ax_1.set_xlim(0, xlim)

    # 修正 title 寫法（保持與您原本一致，但使用 ax_1 設定更安全）
    ax_1.set_title(
        f"Training convergence ({train_method} with {processor}) -- {train_model} \n "
        f"the best test loss : {best_loss:.5f}@{best_epoch}th epoch",
        fontdict=font_title)
    ax_1.set_xlabel('epoch', fontdict=font)
    ax_1.set_ylabel('loss', fontdict=font)
    ax_1.legend(fontsize='x-large')
    training_param = (f"No Epochs: {info_train[0]}; Learning Rate: {info_train[1]}; "
                      f"Momentum: {info_train[2]}; Weight Decay: {info_train[3]}")
    ax_1.text(0.05, 0.95, training_param, transform=ax_1.transAxes,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if save_plot:
        profile_plot = os.path.join(dir_training_records, f"{timestamp}_Training_convergence_plot_Model_{train_model}_Training_Method_{train_method}.png")
        fig_1.savefig(profile_plot, bbox_inches='tight')

    fig_1.canvas.draw()
    fig_1.canvas.flush_events()
    if show_training_plot:
        plt.show(block=False)

    # plot the statistical histogram of learning time in terms of epoch and sample
def lt_plot(lt_epoch, lt_sample, overall_time, info_train, dir_training_records, train_model, train_method, processor, save_plot=False):
    from math import ceil, floor
    import time
    # ----- training time statistics in terms of epoch
    # lt_epoch[0] -= lt_sample[0]
    learning_time_epoch = np.array(lt_epoch)
    mean_lt_epoch = np.mean(learning_time_epoch)
    max_lt_epoch = np.amax(learning_time_epoch)
    min_lt_epoch = np.amin(learning_time_epoch)
    print(
        "mean learning time per epoch: {:.3f} s, maximum epoch learning time: {:.3f} s, minimum epoch learning time: {:.3f} s".
        format(mean_lt_epoch, max_lt_epoch, min_lt_epoch))

    # ----- training time statistics in terms of sample
    lt_sample.sort(reverse=True)
    nex = ceil(0.001 * len(lt_sample))
    learning_time_sample = np.array(lt_sample[nex: -nex])
    mean_lt_sample = np.mean(learning_time_sample).tolist()
    max_lt_sample = np.amax(learning_time_sample).tolist()
    min_lt_sample = np.amin(learning_time_sample).tolist()
    print(
        "mean learning time per sample: {:.3f} s, maximum sample learning time: {:.3f} s, minimum sample learning time: {:.3f} s".
        format(mean_lt_sample, max_lt_sample, min_lt_sample))

    fig_2, axh = plt.subplots(1, 2, figsize=(14, 6))
    time_used = time.strftime("%H:%M:%S", time.gmtime(ceil(overall_time)))
    fig_2.suptitle(f"Training Time Statistics ({train_method} with {processor}) -- {train_model} \n "
                   f"Overall training time : {time_used} ({overall_time:.2f} sec.)",
                   fontsize=20, fontweight='normal')
    axh[0].set_ylabel('no. of epochs', fontdict=font)
    axh[0].set_xlabel('Mean time for training an epoch, sec.', fontdict=font)
    cf = 0.9 * min_lt_epoch
    cc = 1.1 * max_lt_epoch
    bins_epochs_time = np.arange(cf, cc, (cc - cf) / 30)
    axh[0].hist(learning_time_epoch, bins=bins_epochs_time.tolist())
    axh[0].tick_params(axis='both', labelsize='large')
    props = dict(boxstyle='round', facecolor='wheat')
    text_str_0 = (" mean time: %.4f sec. \n max time: %.4f sec. \n min time: %.4f sec. "
                  % (float(mean_lt_epoch), float(max_lt_epoch), float(min_lt_epoch)))
    axh[0].text(0.55, 0.85, text_str_0, transform=axh[0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    axh[1].set_ylabel('no. of batches', fontdict=font)
    axh[1].set_xlabel('Mean time for training a batch sample in an epoch, sec.', fontdict=font)
    sf = 0.9 * min_lt_sample
    sc = 1.1 * max_lt_sample
    bins_samples_time = np.arange(sf, sc, (sc - sf) / 30)
    axh[1].hist(learning_time_sample, bins=bins_samples_time.tolist())
    # axh[1].hist(learning_time_sample, bins=(0.01 * np.array(list(range(101)))).tolist())
    axh[1].tick_params(axis='both', labelsize='large')
    props = dict(boxstyle='round', facecolor='wheat')
    text_str_1 = f"mean time: {mean_lt_sample:.4f} sec. \n max time: {max_lt_sample:.4f} sec. \n min time: {min_lt_sample:.4f} sec."
    axh[1].text(0.55, 0.85, text_str_1, transform=axh[1].transAxes, fontsize=10, verticalalignment='top', bbox=props)

    fig_2.canvas.draw()
    fig_2.canvas.flush_events()
    plt.show(block=False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    training_time_file = os.path.join(dir_training_records, f"{timestamp}_Training_time_Model_{train_model}_Training_Method_{train_method}.png")
    if save_plot:
        fig_2.savefig(str(training_time_file))

    # plt.clf()
