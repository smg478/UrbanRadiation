#=======================================================================================================================
# This file generates an ensemble prediction using outputs from 06, 07, 08 scripts
# predictions will be saved in wdata/submits folder
# threshold = 4 would be valid choice according to out-of-fold predictions
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

from collections import Counter

import numpy as np
import pandas as pd
import sys
#=======================================================================================================================
# bash test.sh /data/testing/ solution.csv

test_folder = '/data/testing/'
solution_fn = 'solution.csv'
wdata_dir = '/wdata/'
if len(sys.argv) > 1:
    test_folder = sys.argv[1]
    solution_fn = sys.argv[2]
#======================================================================================================================
# folder name to save submits
expt_name = 'ann'
sub_dir = wdata_dir + 'submits/' + expt_name + '/'
threshold = 4  # oof best 4


# df1 = pd.read_csv(sub_dir + 'solution_{}_th3_seg0.5_test.csv'.format(expt_name))
# df2 = pd.read_csv(sub_dir + 'solution_{}_th5_seg0.5_test.csv'.format(expt_name))
# df3 = pd.read_csv(sub_dir + 'solution_{}_th7_seg0.5_test.csv'.format(expt_name))

df4 = pd.read_csv(sub_dir + 'solution_{}_th3_seg1.25_test.csv'.format(expt_name))
df5 = pd.read_csv(sub_dir + 'solution_{}_th5_seg1.25_test.csv'.format(expt_name))
df6 = pd.read_csv(sub_dir + 'solution_{}_th7_seg1.25_test.csv'.format(expt_name))

# df7 = pd.read_csv(sub_dir + 'solution_{}_th3_seg0.33_test.csv'.format(expt_name))
# df8 = pd.read_csv(sub_dir + 'solution_{}_th5_seg0.33_test.csv'.format(expt_name))
# df9 = pd.read_csv(sub_dir + 'solution_{}_th7_seg0.33_test.csv'.format(expt_name))

df10 = pd.read_csv(sub_dir + 'solution_{}_th3_seg1500_test.csv'.format(expt_name))
df11 = pd.read_csv(sub_dir + 'solution_{}_th5_seg1500_test.csv'.format(expt_name))
df12 = pd.read_csv(sub_dir + 'solution_{}_th7_seg1500_test.csv'.format(expt_name))

df13 = pd.read_csv(sub_dir + 'solution_{}_th3_seg3000_test.csv'.format(expt_name))
df14 = pd.read_csv(sub_dir + 'solution_{}_th5_seg3000_test.csv'.format(expt_name))
df15 = pd.read_csv(sub_dir + 'solution_{}_th7_seg3000_test.csv'.format(expt_name))

ids = df4.RunID
sources = df4.SourceID

df_combined = pd.DataFrame()
df_combined['RunID'] = ids

# sid1 = df1.SourceID.values
# sid2 = df2.SourceID.values
# sid3 = df3.SourceID.values
sid4 = df4.SourceID.values
sid5 = df5.SourceID.values
sid6 = df6.SourceID.values
# sid7 = df7.SourceID.values
# sid8 = df8.SourceID.values
# sid9 = df9.SourceID.values
sid10 = df10.SourceID.values
sid11 = df11.SourceID.values
sid12 = df12.SourceID.values
sid13 = df13.SourceID.values
sid14 = df14.SourceID.values
sid15 = df15.SourceID.values

# time1 = df1.SourceTime.values
# time2 = df2.SourceTime.values
# time3 = df3.SourceTime.values
time4 = df4.SourceTime.values
time5 = df5.SourceTime.values
time6 = df6.SourceTime.values
# time7 = df7.SourceTime.values
# time8 = df8.SourceTime.values
# time9 = df9.SourceTime.values
time10 = df10.SourceTime.values
time11 = df11.SourceTime.values
time12 = df12.SourceTime.values
time13 = df13.SourceTime.values
time14 = df14.SourceTime.values
time15 = df15.SourceTime.values

np_sid = np.array([#sid1, sid2, sid3,
                   sid4, sid5, sid6,
                   #sid7, sid8, sid9,
                   sid10, sid11, sid12,
                   sid13, sid14, sid15
                   ]).T
np_time = np.array([#time1, time2, time3,
                    time4, time5, time6,
                    #time7, time8, time9,
                    time10, time11, time12,
                    time13, time14, time15
                    , ], dtype=np.float16).T

run_id = []
filtered_label = []
filtered_time = []

for i, rid in enumerate(ids):
    all_labels = np_sid[i]
    all_timess = np_time[i]

    # Count the frequency of non-zero preds
    try:
        preds_nonzero = [x for x in all_labels if x > 0]
        most_common, freq_most_common = Counter(preds_nonzero).most_common(1)[0]
    except:
        freq_most_common = 0

    #most_common, freq_most_common = Counter(all_labels).most_common(1)[0]

    if freq_most_common >= threshold:
        idx = np.where(all_labels==most_common)
        _time = all_timess[idx]
        avg_time = sum(_time)/len(_time)

        run_id.append(rid)
        filtered_label.append(most_common)
        filtered_time.append(avg_time)
        #print('done')

    else:
        run_id.append(rid)
        filtered_label.append(0)
        filtered_time.append(0)

sub = pd.DataFrame()
sub["RunID"] = run_id
sub['SourceID'] = filtered_label
sub["SourceTime"] = filtered_time

print(sub['SourceID'].astype(bool).sum(axis=0))

sub.to_csv(sub_dir + "{}_3tta_th{}_test.csv".format(expt_name, threshold), index=False)


print('done')
