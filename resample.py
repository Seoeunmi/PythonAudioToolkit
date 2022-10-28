import multiprocessing
import argparse
import librosa
import util
import parmap
import os


def resample(input_file, resample_rate, output_path, input_path):
    x, orig_sr = util.read_audio_file(input_file)
    x = librosa.resample(x, orig_sr, resample_rate)
    util.write_audio_file(os.path.join(output_path, util.remove_base_path(input_file, input_path)), x, resample_rate)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='resample.py arguments')
    parser.add_argument('-i', '-I', '--input_path', type=str, required=True, help='input file(directory) path')
    parser.add_argument('-o', '-O', '--output_path', type=str, help='output file(directory) path', default='./')
    parser.add_argument('-r', '-R', '--resample_rate', type=int, required=True, help='target sampling rate')
    parser.add_argument('-c', '-C', '--cores', type=int, help='number of cores', default=multiprocessing.cpu_count())

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    resample_rate = args.resample_rate
    cores = args.cores


    if os.path.isfile(input_path):
        x, orig_sr = util.read_audio_file(input_path)
        x = librosa.resample(y=x, orig_sr=orig_sr, target_sr=resample_rate)
        util.write_audio_file(os.path.join(output_path, os.path.basename(input_path)), x, resample_rate)
    else:
        file_list = util.read_path_list(input_path, extension='wav')
        parmap.map(resample, file_list, resample_rate, output_path, input_path, pm_pbar=True, pm_processes=min(cores, len(file_list)))
