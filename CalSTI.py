# -*- coding: utf-8 -*-
"""
@file      :  CalSTI.py
@Time      :  2022/11/13 17:44
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
import scipy.signal
import torch
import time
import pandas as pd
from moviepy.editor import AudioFileClip
import numpy as np
from torchaudio.transforms import Resample
from pathlib import Path
from model.FpnAttentionEncoder import AttentionFPN
from wav2spec import wave2spec

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("save_model")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


model_path = relative_to_assets('19epoch_attention_STI_FPN_Encoder.pt')
cut_time = 200

def check_audio(wave_data, framerate):
    if framerate != 16000:
        wave_data = scipy.signal.resample(wave_data, int(len(wave_data)/framerate*16000))
    nframes, nchannels = wave_data.shape
    if nchannels == 2:
        wave_data = wave_data.T[0]
    if isinstance(wave_data[0], np.float):
        wave_data = np.array(wave_data * 32768.0, dtype=np.int16)
    elif isinstance(wave_data[0], np.int32):
        wave_data = (wave_data >> 16).astype(np.int16)
    audio_time = len(wave_data) / 16000
    return wave_data, audio_time, 16000


def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(save_model)

    return model


def second_to_time(a):
    if '.5' in str(a):
        ms = 5000
        a = int(a - 0.5)
    else:
        ms = 0000
        a = int(a)
    h = a // 3600
    m = a // 60 % 60
    s = a % 60
    return str("{:0>2}:{:0>2}:{:0>2}.{:0>4}".format(h, m, s, ms))


class ResultPrinter(object):
    def __init__(self):
        self.out_dict = {'start': [], 'end': [], 'STI': []}

    def print_results(self, out):
        already_time = self.get_start_len()
        for i in range(len(out)):
            start = (already_time + i) * 0.5
            end = start + 4
            print('time: [%.1f]s ~ [%.1f]s      STI: [%.4f] ' % (start, end, float(out[i])))
            self.out_dict['start'].append(start)
            self.out_dict['end'].append(end)
            self.out_dict['STI'].append(float(out[i]))

    def save_csv_result(self, save_file_path):
        out_data = pd.DataFrame.from_dict(self.out_dict)
        out_data.to_csv(save_file_path)
        print('Save csv already! Path: ', save_file_path)

    def get_start_len(self):
        return len(self.out_dict['start'])

    def get_csv_result(self):
        return pd.DataFrame.from_dict(self.out_dict)


def get_offline_res(wave_data, audio_time, framerate, model):
    print('Load Model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AttentionFPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)

    # check model
    start_time = time.time()
    net = load_checkpoint(model_path, 99, net, device)
    print('Successfully Loaded model: {}'.format(model_path))
    print('Finished Initialization in {:.3f}s.\n'.format(time.time() - start_time))
    resultPrinter = ResultPrinter()
    net.to(device)

    if audio_time <= cut_time:
        # 25s 以内的音频
        images, empty_index = wave2spec(wave_data, audio_time, framerate, model)  # tensor:[chunk, 3, 224, 224]
        print('Available chunks in total: [ ', images.shape[0], ' ] \n')

        with torch.no_grad():
            net.eval()
            images_reshape = images.to(torch.float32).to(device)
            output_pts = net(images_reshape, [1 for _ in range(images_reshape.shape[0])])
            output_pts = output_pts.squeeze().tolist()

        if empty_index:
            for item in empty_index:
                output_pts.insert(item, 'NaN')
        print('STI Results:')
        if isinstance(output_pts, float):
            output_pts = [output_pts]
        resultPrinter.print_results(out=output_pts)
        return resultPrinter.get_csv_result()


def calSTI(path):
    """
    :param path: mp4 文件的地址
    :return: pd.DataFrame
    """
    my_audio_clip = AudioFileClip(path)
    audio = my_audio_clip.to_soundarray()
    fs = my_audio_clip.fps
    audio, audio_time, fs = check_audio(audio, fs)
    csv_res = get_offline_res(audio, audio_time, fs, 'attention')
    return csv_res


if __name__ == "__main__":
    path = "/Users/bajianxiang/Desktop/你有毒吧.mp4"
    calSTI(path)
