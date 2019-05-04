#=======================================================================================================================
# This file creates smaller segments from training files
# For data without source, equally spaced windows were selected. Window width is: number of counts/30
# First 30 sec was omitted from training
# For data with source, 7 nearest windows from source were generated.
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================
import os
import sys
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

########################################################################################################################
expt_name = 'training_slice'
train_folder = '/data/training/'
train_answr = '/data/trainingAnswers.csv'
wdata_dir = '/wdata/'
if len(sys.argv) > 1:
    train_folder = sys.argv[1]
    train_answr = sys.argv[2]
########################################################################################################################

target_df = pd.read_csv(train_answr)
train_ids = target_df.RunID
source_ids = target_df.SourceID
source_times = target_df.SourceTime * 1000000  # micro-sec -> sec
answer = []
answer_fn = []

# create directory to save files
slice_data_dir = wdata_dir + expt_name + '/'
if not os.path.exists(slice_data_dir):
    os.makedirs(slice_data_dir)

#=======================================================================================================================
# loop over each training file
for i, id in enumerate(tqdm(train_ids)):
    source_id = source_ids[i]
    source_time = source_times[i]

    df = pd.read_csv(train_folder + '{}.csv'.format(id))
    time = df[df.columns[0]]
    length = len(time)
    df['time_cumsum'] = np.array(time.cumsum(), dtype=np.float32)

    # divide test file into 30 segments
    num_seg = 30
    seg_width = int(length / num_seg)
    if seg_width > 10000: seg_width = 10000
    if seg_width < 2000: seg_width = 2000

    if source_id == 0:
        # find 10 equally spaced windows
        num_windows = 15
        f = lambda m, n: [k * n // m + n // (2 * m) for k in range(m)]
        _idxs = f(num_windows, length)

        # find 30 sec index
        df_sort = df.ix[(df['time_cumsum'] - 30000000).abs().argsort()[:2]]
        idx30 = df_sort.index.tolist()[0]

        # index without first 30 sec
        _idxs = [k for k in _idxs if k >= idx30]

        for k, j in enumerate(_idxs):  # loop over windows over entire file
            type = ['fixed', 'variable']
            random_type = random.choice(type)

            if random_type == 'fixed':
                items_a = [3000, 1500]
                xa = random.choice(items_a)

                start = int(j - xa)
                end = int(j + xa)
                if j - xa < 0: start = 0
                if j + xa > length: end = length

            else:
                items_b = [0.33, 0.5, 0.75, 1, 1.25]
                xb = random.choice(items_b)

                start = int(j - seg_width * xb)
                end = int(j + seg_width * xb)
                if j - seg_width * xb < 0: start = 0
                if j + seg_width * xb > length: end = length

            df_n = df[start:end]
            _id = str(id) + '_' + str(k)
            df_n.to_csv(wdata_dir + expt_name + '/{}.csv'.format(_id), index=False)
            answer.append(source_id)
            answer_fn.append(_id)

    else:
        # find the closest time as described in training anewers file
        df_sort = df.ix[(df['time_cumsum'] - source_time).abs().argsort()[:2]]
        idx = df_sort.index.tolist()[0]
        df = df.drop('time_cumsum', 1)

        # seg 1
        start = int(idx - seg_width* 0.5)
        end = int(idx +  seg_width* 0.5)
        if idx -  seg_width * 0.5 < 0: start = 0
        if idx +  seg_width* 0.5 > length: end = length
        df_a = df[start:end]
        id_a = str(id) + '_' + str(1)
        df_a.to_csv(wdata_dir + expt_name + '/{}.csv'.format(id_a), index=False)
        answer.append(source_id)
        answer_fn.append(id_a)

        # seg 2
        start = int(idx - seg_width* 0.25)
        end = int(idx + seg_width* 0.75)
        if idx - seg_width* 0.25 < 0: start = 0
        if idx + seg_width* 0.75 > length: end = length
        df_b = df[start:end]
        id_b = str(id) + '_' + str(2)
        df_b.to_csv(wdata_dir + expt_name +'/{}.csv'.format(id_b), index=False)
        answer.append(source_id)
        answer_fn.append(id_b)

        # seg 3
        start = int(idx - seg_width* 0.75)
        end = int(idx + seg_width* 0.25)
        if idx - seg_width* 0.75 < 0: start = 0
        if idx + seg_width* 0.25 > length: end = length
        df_c = df[start:end]
        id_c = str(id) + '_' + str(3)
        df_c.to_csv(wdata_dir + expt_name+'/{}.csv'.format(id_c), index=False)
        answer.append(source_id)
        answer_fn.append(id_c)

        # seg 4
        start = int(idx - seg_width * 0.33)
        end = int(idx +  seg_width * 0.33)
        if idx -  seg_width * 0.33 < 0: start = 0
        if idx +  seg_width * 0.33 > length: end = length
        df_d = df[start:end]
        id_d = str(id) + '_' + str(4)
        df_d.to_csv(wdata_dir + expt_name +'/{}.csv'.format(id_d), index=False)
        answer.append(source_id)
        answer_fn.append(id_d)

        # seg 5
        start = int(idx - seg_width * 1.25)
        end = int(idx +  seg_width * 1.25)
        if idx -  seg_width*1.25 < 0: start = 0
        if idx +  seg_width*1.25 > length: end = length
        df_e = df[start:end]
        id_e = str(id) + '_' + str(5)
        df_e.to_csv(wdata_dir + expt_name + '/{}.csv'.format(id_e), index=False)
        answer.append(source_id)
        answer_fn.append(id_e)

        # seg 6
        start = int(idx - 3000)
        end = int(idx +  3000)
        if idx -  3000 < 0: start = 0
        if idx +  3000 > length: end = length
        df_e = df[start:end]
        id_e = str(id) + '_' + str(6)
        df_e.to_csv(wdata_dir + expt_name + '/{}.csv'.format(id_e), index=False)
        answer.append(source_id)
        answer_fn.append(id_e)

        # seg 7
        start = int(idx - 1500)
        end = int(idx +  1500)
        if idx -  1500 < 0: start = 0
        if idx +  1500 > length: end = length
        df_e = df[start:end]
        id_e = str(id) + '_' + str(7)
        df_e.to_csv(wdata_dir + expt_name + '/{}.csv'.format(id_e), index=False)
        answer.append(source_id)
        answer_fn.append(id_e)

df_train_ans_slice = pd.DataFrame()
df_train_ans_slice['RunID'] = answer_fn
df_train_ans_slice['SourceID'] = answer

df_train_ans_slice.to_csv(wdata_dir + 'trainingAnswers_slice.csv', index=False)

print('done')
