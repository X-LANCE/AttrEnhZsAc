import pypeln.process as pr
import argparse
import os
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=20, help='number of workers')
parser.add_argument('-v', '--video_path', type=str, help='path of video data')
parser.add_argument('-o', '--output_path', type=str, help='path of output wav data')
parser.add_argument('--sr', type=int, default=32000, help='sample rate')
args = parser.parse_args()

print(f"""
Converting .mp4 in {args.video_path} \
to .wav in {args.output_path} \
with sample rate {args.sr}. \
""")


mp4_to_wav = 'ffmpeg -n -i {} -acodec pcm_s16le -ac 1 -ar {} {} -loglevel quiet'
# adapoted from https://github.com/hche11/VGGSound/blob/master/preprocess_audio.py

os.makedirs(args.output_path, exist_ok=True)
videos = glob(os.path.join(args.video_path, '*.mp4'))
sr = args.sr

def converse_mp4(file):
    file_name = os.path.basename(file).split('.')[0]
    out_path = os.path.join(args.output_path, f'{file_name}.wav')
    flag = os.system(mp4_to_wav.format(file, sr, out_path))
    return file_name, flag


with tqdm(total=len(videos), desc='Converse mp4 to wav') as pbar:
    for file_name, flag in pr.map(converse_mp4, videos, workers=args.c, maxsize=2*args.c):
        if flag:
            print(f"Failed to convert {file_name}")
        pbar.update()



