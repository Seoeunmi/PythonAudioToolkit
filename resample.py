import os, time, multiprocessing, argparse
import librosa
from p_tqdm import p_umap
from functools import partial
import util


def resample(input_file, resample_rate, output_path, input_path):
    x, orig_sr = util.read_audio_file(input_file)
    x = librosa.resample(x, orig_sr, resample_rate)
    util.write_audio_file(os.path.join(output_path, util.remove_base_path(input_file, input_path)), x, resample_rate)


if __name__=='__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='resample.py arguments')
    parser.add_argument('-i', '-I', '--input_path', type=str, required=True, help='input file(directory) path')
    parser.add_argument('-o', '-O', '--output_path', type=str, help='output file(directory) path', default='./')
    parser.add_argument('-r', '-R', '--resample_rate', type=int, required=True, help='target sampling rate')
    parser.add_argument('-c', '-C', '--cores', type=int, help='number of cores', default=multiprocessing.cpu_count())
    parser.add_argument('-m', '-M', '--core_multiplier', type=float, help='multiply value to cores', default=1)

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    resample_rate = args.resample_rate
    cores = max(1, int(args.cores*args.core_multiplier))


    if os.path.isfile(input_path):
        x, orig_sr = util.read_audio_file(input_path)
        x = librosa.resample(y=x, orig_sr=orig_sr, target_sr=resample_rate)
        util.write_audio_file(os.path.join(output_path, os.path.basename(input_path)), x, resample_rate)
    else:
        file_list = util.read_path_list(input_path, extension='wav')
        p_umap(partial(resample, resample_rate=resample_rate, output_path=output_path, input_path=input_path),
               file_list, num_cpus=min(cores, len(file_list)), desc='Process')
    print(f"Finished! {util.second_to_dhms_string(time.time()-start)} time taken.")
