#=======================================================================================================================
# This file creates features for model treaining, from segmented training files
# 100 features from a 100 bin spectrum that covers 0-3000keV
# additional 51 features from peak to compton ration and peak to peak ratio using peaks associated with a source
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import numpy as np
import sys
import pandas as pd
from tqdm import tqdm


# bash train.sh /data/training/ /data/trainingAnswers.csv
########################################################################################################################
train_folder = '/data/training/'
train_answr = '/data/trainingAnswers.csv'
wdata_dir = '/wdata/'
if len(sys.argv) > 1:
    train_folder = sys.argv[1]
    train_answr = sys.argv[2]
########################################################################################################################

target_df = pd.read_csv(wdata_dir + 'trainingAnswers_slice.csv')
train_ids = target_df.RunID

bin_size = 30
bins = np.arange(0, 3000, bin_size)

train_bin = []

for i, id in enumerate(tqdm(train_ids)):
    df = pd.read_csv(wdata_dir + 'training_slice/{}.csv'.format(id))
    energy = df[df.columns[1]]

    out = pd.cut(energy, bins=bins, include_lowest=True)
    counts = out.value_counts(sort=False)

    np_counts = np.array(counts.values, dtype=np.float32)
    np_counts = np_counts / np.sum(np_counts)

    # ===================================================================================================================
    # peak to compton ratio feature
    # ===================================================================================================================

    count1 = np_counts[1] / np.sum(np_counts[0:1])
    count2 = np_counts[2] / np.sum(np_counts[0:2])
    count3 = np_counts[3] / np.sum(np_counts[0:2])
    count4 = np_counts[4] / np.sum(np_counts[0:3])
    count5 = np_counts[5] / np.sum(np_counts[0:4])
    count6 = np_counts[6] / np.sum(np_counts[0:5])
    count7 = np_counts[7] / np.sum(np_counts[0:6])
    count8 = np_counts[8] / np.sum(np_counts[0:7])
    count9 = np_counts[9] / np.sum(np_counts[0:8])
    count10 = np_counts[10] / np.sum(np_counts[0:9])
    count11 = np_counts[11] / np.sum(np_counts[0:10])
    count12 = np_counts[12] / np.sum(np_counts[0:11])
    count13 = np_counts[13] / np.sum(np_counts[0:12])
    count14 = np_counts[14] / np.sum(np_counts[0:13])
    count15 = np_counts[15] / np.sum(np_counts[0:14])
    count16 = np_counts[16] / np.sum(np_counts[0:15])
    count17 = np_counts[17] / np.sum(np_counts[0:16])
    count18 = np_counts[18] / np.sum(np_counts[0:17])

    count19 = np_counts[19] / np.sum(np_counts[0:18])
    count20 = np_counts[20] / np.sum(np_counts[0:19])
    count21 = np_counts[21] / np.sum(np_counts[0:20])
    count22 = np_counts[87] / np.sum(np_counts[0:86])

    np_counts_peaks = np.array(
        [count1, count2, count3, count4, count5, count6, count7, count8, count9, count10,
         count11, count12, count13, count14, count15, count16, count17, count18, count19, count20,
         count21, count22]).T

    # ===================================================================================================================
    # peak to peak ratio feature
    # ===================================================================================================================
    # HEU
    ratio1 = np_counts[0] / (np_counts[17] + np_counts[19])
    ratio2 = np_counts[3] / (np_counts[17] + np_counts[19])
    ratio3 = np_counts[6] / (np_counts[17] + np_counts[19])
    ratio4 = np_counts[87] / (np_counts[17] + np_counts[19])

    # WGPu
    ratio5 = np_counts[1] / (np_counts[12] + np_counts[13])
    ratio6 = np_counts[2] / (np_counts[12] + np_counts[13])
    ratio7 = np_counts[3] / (np_counts[12] + np_counts[13])
    ratio8 = np_counts[6] / (np_counts[12] + np_counts[13])
    ratio9 = np_counts[21] / (np_counts[12] + np_counts[13])

    # I-131
    ratio10 = np_counts[1] / np_counts[12]
    ratio11 = np_counts[2] / np_counts[12]
    ratio12 = np_counts[6] / np_counts[12]
    ratio13 = np_counts[9] / np_counts[12]
    ratio14 = np_counts[21] / np_counts[12]

    # Tc-99m
    ratio15 = np_counts[0] / np_counts[4]
    ratio16 = np_counts[1] / np_counts[4]
    ratio17 = np_counts[10] / np_counts[4]

    # HEU + Tc-99m
    ratio18 = np.sum(np_counts[0:7]) / np.sum(np_counts[0:21])
    ratio19 = ratio1 / ratio15
    ratio20 = ratio2 / ratio15
    ratio21 = ratio3 / ratio15
    ratio22 = ratio4 / ratio15

    ratio23 = ratio1 / ratio16
    ratio24 = ratio2 / ratio16
    ratio25 = ratio3 / ratio16
    ratio26 = ratio4 / ratio16

    ratio27 = ratio1 / ratio17
    ratio28 = ratio2 / ratio17
    ratio29 = ratio3 / ratio17
    ratio30 = ratio4 / ratio17

    np_ratio_peaks = np.array(
        [ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7, ratio8, ratio9, ratio10,
         ratio11, ratio12, ratio13, ratio14, ratio15, ratio16, ratio17, ratio18, ratio19, ratio20,
         ratio21, ratio22, ratio23, ratio24, ratio25, ratio26, ratio27, ratio28, ratio29, ratio30]).T

    # ===================================================================================================================
    np_id = np.expand_dims(id, axis=1)
    np_target = np.expand_dims(target_df.SourceID[i], axis=1)

    np_id_counts_target = np.concatenate([np_id, np_counts, np_counts_peaks, np_ratio_peaks, np_target], axis=0)
    train_bin.append(np_id_counts_target)

    # ax = out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(6, 4))
    # ax.set_yscale("log", nonposy='clip')
    # ax.set_xticklabels(np.arange(0, 3000, 500))
    # #ax.set_xticklabels([c[1:-1].replace(",", " to") for c in out.cat.categories])
    # plt.show()

df_train_bin = pd.DataFrame(train_bin)
df_train_bin.to_csv(wdata_dir + 'train_feature_bin_{}_slice.csv'.format(bin_size), index=False)
print('done')
