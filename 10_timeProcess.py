#=======================================================================================================================
# This file fine-tunes source location using outputs from 09 script
# predictions will be saved in wdata/submits folder and also in current directory
# Search for highest number of counts for associated peaks in 1.5*segment_length range, use seed time from 09 script,
# where, segment lenght = number of counts in test file / 30
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================

import gc
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from sklearn.externals import joblib


#=======================================================================================================================
def find_counts(en, src_id):

    # convert energy values to bin counts and normalize
    out = pd.cut(en, bins=bins, include_lowest=True)
    counts = out.value_counts(sort=False)
    np_counts = np.array(counts.values, dtype=np.float32)
    np_counts = np_counts / np.sum(np_counts)


    np_counts = np_counts.reshape(1, 99)
    np_counts = scaler.transform(np_counts)
    np_counts = np.squeeze(np_counts)

    if src_id == 1: # HEU
        # HEU
        count1 = np_counts[0]
        count2 = np_counts[3]
        count3 = np_counts[6]
        count4 = np_counts[17]
        count5 = np_counts[19]
        count6 = np_counts[87]

        total_counts = count1 + count2 + count3 + count4 + count5 + count6
        #total_counts = count4 + count5 + count6

    elif src_id == 2: # WPu 77, 407, 653 // 33, 61, 101, 207, 381
        count1 = np_counts[1]
        count2 = np_counts[2]
        count3 = np_counts[3]
        count4 = np_counts[6]
        count5 = np_counts[12]
        count6 = np_counts[13]
        count7 = np_counts[21]

        total_counts = count1 + count2 + count3 + count4 + count5 + count6 + count7
        #total_counts = count5 + count6 + count7

    elif src_id == 3: # I
        count1 = np_counts[1]
        count2 = np_counts[2]
        count3 = np_counts[6]
        count4 = np_counts[9]
        count5 = np_counts[12]
        count6 = np_counts[21]
        total_counts = count1 + count2 + count3 + count4 + count5 + count6
        #total_counts = count4 + count5 + count6

    elif src_id == 4: # Co
        count1 = np_counts[39]
        count2 = np_counts[43]
        count3 = np_counts[44]
        total_counts = count1 + count2 + count3

    elif src_id == 5: # Tc
        count1 = np_counts[0]
        count2 = np_counts[1]
        count3 = np_counts[4]
        count4 = np_counts[10]
        total_counts = count1 + count2 + count3 + count4

    elif src_id == 6: # HEU+Tc
        count1 = np_counts[0]
        count2 = np_counts[3]
        count3 = np_counts[6]
        count4 = np_counts[17]
        count5 = np_counts[19]
        count6 = np_counts[87]

        count7 = np_counts[1]
        count8 = np_counts[4]
        count9 = np_counts[10]

        total_counts = count1 + count2 + count3 + count4 + count5 + count6 + count7 + count8 + count9
    else:
        total_counts = 0

    return total_counts


# A function that can be called to do work:
def work(arg):
    # Split the list to individual variables:
    i, id = arg
    id = int(id)
    source_id = source_ids[i]
    apprx_time = coarse_time[i]

    if source_id == 0:
        return id,source_id, 0

    else:
        #df = pd.read_csv(data_dir + 'testing/{}.csv'.format(id))
        df = pd.read_csv(test_folder + '{}.csv'.format(id))
        time = df[df.columns[0]]
        energy = df[df.columns[1]]
        df['time_cumsum'] = np.array(time.cumsum(), dtype=np.float32)
        length = len(time)

        #=====================================================================
        # divide test file into 30 segments to get unit segment size
        seg_width = int(length / 30)
        if seg_width > 20000: seg_width = 20000
        if seg_width < 1000: seg_width = 1000
        # find 30 sec index
        df_sort = df.ix[(df['time_cumsum'] - 30000000).abs().argsort()[:2]]
        idx30 = df_sort.index.tolist()[0]

        # find apprx time index
        df_sort_apprx = df.ix[(df['time_cumsum'] - apprx_time*1000000).abs().argsort()[:2]]
        idx_apprx = df_sort_apprx.index.tolist()[0]
        #===================================================================

        #----------------------------------------------------------------------------------
        #                       center of scan window
        #                                  /
        #---------------|--------|---------/----------|----------|--------------------------
        #               |        |         /          |          |
        # --------------|--------|---------/----------|----------|--------------------------
        #                                  /
        #========================|  1000 data points  |=====================================
        #===============|           1 segment data points        |==========================
        #-----------------------------------------------------------------------------------

        # average time from different segment sizes (range 0f scan)
        #seg_mul = [0.25, 0.5, 0.75]             # 93.14 //
        seg_mul = [0.75]
        #seg_mul = [1.5]
        preds = 0
        for k in seg_mul:  # search range
            start = idx_apprx - seg_width*k
            end = idx_apprx + seg_width*k
            if start < idx30: start = idx30
            if end > length-seg_width*k: end = length-seg_width*k

            for j in range (int(start), int(end), propagation_step):
                # # resolution 1
                en1 = energy[j - 500 : j + 500]
                total_cnts1 = find_counts(en1, source_id)

                # resolution 2
                en2 = energy[j - int(seg_width * 0.25) : j + int(seg_width * 0.25)]  # window for max count
                total_cnts2 = find_counts(en2, source_id)

                # resolution 3
                en3 = energy[j - int(seg_width * 0.5) : j + int(seg_width * 0.5)]  # window for max count
                total_cnts3 = find_counts(en3, source_id)

                # resolution 4
                en4 = energy[j - int(seg_width * 0.66) : j + int(seg_width * 0.66)]  # window for max count
                total_cnts4 = find_counts(en4, source_id)

                df.at[j, 'peak_counts_{}'.format(k)] = total_cnts1 + total_cnts2 + total_cnts3 + total_cnts4
                #df.at[j, 'peak_counts_{}'.format(k)] = total_cnts3

            _df = df[np.isfinite(df['peak_counts_{}'.format(k)])]
            time_at_max_count = _df.loc[_df['peak_counts_{}'.format(k)].idxmax(), 'time_cumsum']
            pred_t = time_at_max_count/1000000
            preds = preds + pred_t

            del _df
            gc.collect()

        pred = preds/len(seg_mul)

        return id, source_id, pred

#=======================================================================================================================
# Input: output file from model prediction
#=======================================================================================================================
# bash test.sh /data/testing/ solution.csv

test_folder = '/data/testing/'
solution_fn = 'solution.csv'
wdata_dir = '/wdata/'
if len(sys.argv) > 1:
    test_folder = sys.argv[1]
    solution_fn = sys.argv[2]

# folder name to save submits
expt_name = 'ann'
sub_dir = wdata_dir + 'submits/' + expt_name + '/'
df_train = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
#######################################################################################################
# submit without pseudo
input_fn = 'ann_3tta_th4_test.csv'
#######################################################################################################
input_df = pd.read_csv(sub_dir + input_fn )
propagation_step = 100

test_ids = input_df.RunID
source_ids = input_df.SourceID
coarse_time = input_df.SourceTime
#=======================================================================================================================

x_trn = df_train.iloc[:,1:100]
# scale train
X = x_trn.values
where_are_NaNs = np.isnan(X)
where_are_infs = np.isinf(X)
X[where_are_NaNs] = 0
X[where_are_infs] = 0


scaler = RobustScaler()
scaler.fit(X)
scaled_train_X = scaler.transform(X)
X = scaled_train_X

#scaler = joblib.load("scaler.save")
# bins for test segment
bins = np.arange(0,3000,30)

#=======================================================================================================================
# Parallel code
idx_list = list(input_df.index.values)
id_list = test_ids.tolist()
arg_instances =  list(zip(idx_list, id_list))
# parallel processing
results = Parallel(n_jobs=8*2, verbose=50, batch_size=2)(map(delayed(work), arg_instances))
#=======================================================================================================================
# write submission file
df_pred_time = pd.DataFrame(results)
df_pred_time.columns = ["RunID", "SourceID", "SourceTime"]
df_pred_time.to_csv(sub_dir + 'solution_{}_mul75_robust_peakall.csv'.format(input_fn), index = False)
df_pred_time.to_csv(solution_fn, index = False)

print('All done. Final predictions are saved in current directory.')
