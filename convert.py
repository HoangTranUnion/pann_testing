import os
import ffmpeg
import librosa
import soundfile

from pydub import AudioSegment

try:
    os.makedirs('example/16k', mode=0o777)
    os.makedirs('example/8k', mode=0o777)
except FileExistsError:
    pass

for file_path in os.listdir('example/'):
    fp = 'example/' + file_path
    if os.path.isfile(fp):
        audio = AudioSegment.from_file(fp)
        audio.export(f'example/{"".join(file_path.split(".")[0]) + ".wav"}', format='wav')
        new_file_path = f'example/{"".join(file_path.split(".")[0]) + ".wav"}'
        y1, s1 = librosa.load(new_file_path, sr=16000)  # Downsample 44.1kHz to 16kHz
        y2, s2 = librosa.load(new_file_path, sr=8000)

        fp_16k = f'example/16k/{"".join(file_path.split(".")[0]) + "_16k.wav"}'
        fp_8k = f'example/8k/{"".join(file_path.split(".")[0]) + "_8k.wav"}'

        soundfile.write(fp_16k, y1, s1)
        soundfile.write(fp_8k, y2, s2)
