import pycuda.driver as cuda

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


import scipy.io.wavfile
import csv
import math

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from FourierTransform import FourierTransform



"""
The dialect regions are:
     dr1:  New England
     dr2:  Northern
     dr3:  North Midland
     dr4:  South Midland
     dr5:  Southern
     dr6:  New York City
     dr7:  Western
     dr8:  Army Brat (moved around)

       Dialect
      Region(dr)    #Male    #Female    Total
      ----------  --------- ---------  ----------
         1         31 (63%)  18 (27%)   49 (8%)  
         2         71 (70%)  31 (30%)  102 (16%) 
         3         79 (67%)  23 (23%)  102 (16%) 
         4         69 (69%)  31 (31%)  100 (16%) 
         5         62 (63%)  36 (37%)   98 (16%) 
         6         30 (65%)  16 (35%)   46 (7%) 
         7         74 (74%)  26 (26%)  100 (16%) 
         8         22 (67%)  11 (33%)   33 (5%)
       ------     --------- ---------  ---------- 
         8        438 (70%) 192 (30%)  630 (100%)
"""


class TIMIT():
    
    @staticmethod
    def training_directory_loader(pathName, fileName):
        trainingFilePathName = pathName + fileName
        #
        cols = ['dialect_region', 'speaker_id', 'file_index', 'phn', 'wav', 'txt', 'wrd']
        df = pd.DataFrame(columns=cols)
        #
        with open(trainingFilePathName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    filename = row[4]
                    fullPathFileName = row[6]
                    #
                    # find out the file name root  (file name excluding the extension)
                    idx = filename.find('.')
                    filename_root = filename[:idx]
                    file_index = row[2] + "_" + row[3] + "_" + filename_root   # dialect_region + speaker_id + filename_root
                    #
                    r = (df['file_index'] == file_index)
                    if not r.any():
                        rowCnt = df['file_index'].count()
                        df.loc[rowCnt] = [row[2], row[3], file_index, None, None, None, None]
                        r = (df['file_index'] == file_index)
                    #
                    # find out the file type (the last three characters of a file name)
                    l=len(filename)
                    file_type = filename[l-3:l]
                    if file_type == "PHN":   # a phone file
                        df.loc[r, 'phn'] = fullPathFileName
                    elif file_type == "wav":
                        df.loc[r, 'wav'] = fullPathFileName
                    elif file_type == "TXT":
                        df.loc[r, 'txt'] = fullPathFileName
                    elif file_type == "WRD":
                        df.loc[r, 'wrd'] = fullPathFileName
                #
                line_count += 1
            #
            print(f'Processed {line_count} lines.')
        #
        return df
    
    
    # --------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def GenerateTIMITTrainingSetDict():
        TIMITRoot = "C:\\Data\\TIMIT\\"
        trainingFileName = "train_data.csv"
        trainingFilePathName = TIMITRoot + trainingFileName
        trainSet = TIMIT.training_directory_loader(TIMITRoot, trainingFileName)
        
        dataframeFileName = "train_data.df"
        dataframeFilePathName = TIMITRoot + dataframeFileName
        trainSet.to_pickle(dataframeFilePathName)  # save the training data set to a file
        
        trainSet = pd.read_pickle(dataframeFilePathName)
    
    
    # --------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def TIMIT_PHN_reader(filePath, phoneFileName, sample_cnt, frameBitCnt, strideBitCnt):
        """
        
        RETURN:
            frm_phn_xIdx: frame to phone cross index with the following format
            
            [frameIndex, [soundSampleIndex, soundSampleIndex + frameSize], phoneCount, percentageList, phoneList], where
            
                'phoneCount' represents the number phones the frame covers; it is an integer that's equal or greater than 1;
                'percentageList' is a list of percentages representing each phone's coverage within the frame (number of elements equal to 'phoneCount') 
                'phoneList' is a list of phones representing the phones within the frame (number of elements equal to 'phoneCount') 
        """
        #
        phoneFilePathName = filePath + phoneFileName
        phone_list = open(phoneFilePathName, 'r').read().split()
        #
        phnCnt  = int(len(phone_list)/3)
        phn_range  = np.zeros((phnCnt, 2), dtype=int)
        phn_symble = ['  '    for _ in range(0, phnCnt)]
        #
        for i in range(0, phnCnt):
            idx = i*3
            phn_range[i][0] = phone_list[idx]
            phn_range[i][1] = phone_list[idx+1]
            phn_symble[i]   = phone_list[idx+2]
        #
        # duration of each phone
        #phn_range[:,1] - phn_range[:,0]
        #
        frm_phn_xIdx = []
        frmIdx = -1
        #
        for sampleIdx in range(0, sample_cnt-frameBitCnt, strideBitCnt):
            frmIdx += 1
            pS = -1
            pE = -1
            #
            phnIdx=0
            while(phnIdx<phnCnt and pS==-1 and pE==-1):
                if phn_range[phnIdx, 0]<=sampleIdx and sampleIdx<phn_range[phnIdx, 1]:
                    # found the phone index where the frame starts
                    pS = phnIdx
                    #
                    # look for the phone index where the frame ends
                    while(phnIdx<phnCnt and pE==-1):
                        if (sampleIdx+strideBitCnt-1)<phn_range[phnIdx, 1]:
                            # found the phone index where the frame ends
                            pE = phnIdx
                        else:
                            phnIdx += 1
                    #
                else:
                    phnIdx += 1
                #
            #
            if(pS>=0 and pE>=0):
                pct = []
                if pS==pE:
                    pct.append(1.0)
                else: 
                    for p in range(pS, pE+1):
                        sS = frmIdx * strideBitCnt
                        sE = sS + frameBitCnt
                        if p==pS: 
                            pct.append((phn_range[p, 1]-sS) / frameBitCnt)
                        elif p<pE:
                            pct.append((phn_range[p, 1]-phn_range[p, 0]) / frameBitCnt)
                        elif p==pE:
                            pct.append((sE-phn_range[p, 0]) / frameBitCnt)
                    #
                #
                phns = []
                for p in range(pS, pE+1):
                    phns.append(phn_symble[p])
                #
                frm_phn_xIdx.append([frmIdx, [sampleIdx, sampleIdx+frameBitCnt], (pE-pS+1), pct, phns])
            #
        #
        return frm_phn_xIdx
    
    
    # --------------------------------------------------------------------------------------------------------------------------

"""
# testing the frame-phone cross reference

filePath = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR1\\FJSP0\\"
phoneFileName = "SX84.PHN"
waveFileName  = "SX84.WAV.wav"

(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds
channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt = FourierTransform.FFT(filePath+waveFileName, 0, frame_size, frame_stride, duration=0, emphasize=False)

frameBitCnt  = int(sample_rate*frame_size)
strideBitCnt = int(sample_rate*frame_stride)
frm_phn_xIdx = TIMIT_PHN_reader(filePath, phoneFileName, sample_cnt, frameBitCnt, strideBitCnt)
"""

# --------------------------------------------------------------------------------------------------------------------------


# phoncode:

phonecode = [
        [1,  'b'],
        [2,  'd'],
        [3,  'g'],
        [4,  'p'],
        [5,  't'],
        [6,  'k'],
        [7,  'bcl'],
        [8,  'dcl'],
        [9,  'gcl'],
        [10, 'pcl'],
        [11, 'tcl'],
        [12, 'kcl'],  
        [13, 'dx'],
        [14, 'q'],
        
        [15, 'jh'],
        [16, 'ch'],
      
        [17,  's'],
        [18, 'sh'],
        [19,  'z'],
        [20, 'zh'],
        [21,  'f'],
        [22, 'th'],
        [23,  'v'],
        [24, 'dh'],
      
        [25,  'm'],
        [26,  'n'],
        [27, 'ng'],
        [28, 'em'],
        [29, 'en'],
        [30,'eng'],
        [31, 'nx'],
        
        [32,  'l'],
        [33,  'r'],
        [34,  'w'],
        [35,  'y'],
        [36, 'hh'],
        [37, 'hv'],
        [38, 'el'],
        
        [39, 'iy'],
        [40, 'ih'],
        [41, 'eh'],
        [42, 'ey'],
        [43, 'aa'],
        [44, 'ae'],
        [45, 'aw'],
        [46, 'ay'],
        [47, 'ah'],
        [48, 'ao'],
        [49, 'oy'],
        [50, 'ow'],
        [51, 'uh'],
        
        [52,  'uw'],
        [53,  'ux'],
        [54,  'er'],
        [55,  'ax'],
        [56,  'ix'],
        [57, 'axr'],
        [58,'ax-h'],

        [59, 'pau'],
        [60, 'epi'],
        [61, 'h#']]

"""
for i in range(0, len(phoncode)):
    print("'", phoncode[i][1], "' : ", phoncode[i][0], ",", sep='')
"""

phonedict = {  
    'b' : 1,
    'd' : 2,
    'g' : 3,
    'p' : 4,
    't' : 5,
    'k' : 6,
    'bcl' : 7,
    'dcl' : 8,
    'gcl' : 9,
    'pcl' : 10,
    'tcl' : 11,
    'kcl' : 12,
    'dx' : 13,
    'q' : 14,
    'jh' : 15,
    'ch' : 16,
    's' : 17,
    'sh' : 18,
    'z' : 19,
    'zh' : 20,
    'f' : 21,
    'th' : 22,
    'v' : 23,
    'dh' : 24,
    'm' : 25,
    'n' : 26,
    'ng' : 27,
    'em' : 28,
    'en' : 29,
    'eng' : 30,
    'nx' : 31,
    'l' : 32,
    'r' : 33,
    'w' : 34,
    'y' : 35,
    'hh' : 36,
    'hv' : 37,
    'el' : 38,
    'iy' : 39,
    'ih' : 40,
    'eh' : 41,
    'ey' : 42,
    'aa' : 43,
    'ae' : 44,
    'aw' : 45,
    'ay' : 46,
    'ah' : 47,
    'ao' : 48,
    'oy' : 49,
    'ow' : 50,
    'uh' : 51,
    'uw' : 52,
    'ux' : 53,
    'er' : 54,
    'ax' : 55,
    'ix' : 56,
    'axr' : 57,
    'ax-h' : 58,
    'pau' : 59,
    'epi' : 60,
    'h#' : 61 }

