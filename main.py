import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
from model_setup import set_model
from pathlib import Path
import models
import inspect

PANN_PATH = os.path.join(Path.home(), "panns_data")


def print_audio_tagging_result(clipwise_output, file_name):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    with open(file_name, 'w') as f:
        # Print audio tagging top probabilities
        for k in range(10):
            line = '{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
            clipwise_output[sorted_indexes[k]])
            f.write(f'{line}\n')
            print(line)


def plot_sound_event_detection_result(framewise_output, folder, filename):
    """Visualization of sound event detection result.
    Args:
      framewise_output: (time_steps, classes_num)
    """

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    for idx in idxes:
        line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
        lines.append(line)

    plt.legend(handles=lines)
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.ylim(0, 1.)
    full_path = os.path.join(folder, filename)
    plt.savefig(full_path)
    print('Save fig to {}'.format(filename))


def test_audio(model_audio, sample_rate_folder = None, **kwargs):
    if kwargs:
        model = set_model(model_audio, **kwargs)
    else:
        model = set_model(model_audio)
    audio_path = 'example/'
    if sample_rate_folder is not None:
        audio_path += sample_rate_folder + '/'
    cp_path = os.path.join(PANN_PATH, f"{model_audio}.pth")
    if os.path.isfile(cp_path):
        for index, p in enumerate(os.listdir(audio_path)):
            new_audio_path = audio_path + p
            if os.path.isfile(new_audio_path):
                print(f"Testing on file {p}")
                (audio, _) = librosa.core.load(new_audio_path, sr=32000, mono=True)
                audio = audio[None, :]  # (batch_size, segment_samples)

                print('------ Audio tagging ------')
                at = AudioTagging(model = model, checkpoint_path= cp_path, device='cuda')
                (clipwise_output, embedding) = at.inference(audio)

                try:
                    os.makedirs(model_audio)
                except FileExistsError:
                    pass
                print_audio_tagging_result(clipwise_output[0], os.path.join(model_audio, f"{model_audio}_{p.split('.')[0]}.txt"))


def test_sound(model_sound_name, sample_rate_folder = None, **kwargs):
    if kwargs:
        model = set_model(model_sound_name, **kwargs)
    else:
        model = set_model(model_sound_name)

    audio_path = 'example/'
    if sample_rate_folder is not None:
        audio_path += sample_rate_folder + '/'
    cp_path = os.path.join(PANN_PATH, f"{model_sound_name}.pth")
    if os.path.isfile(cp_path):
        for index, p in enumerate(os.listdir(audio_path)):
            new_audio_path = audio_path + p
            if os.path.isfile(new_audio_path):
                print(f"Testing on file {p}")
                (audio, _) = librosa.core.load(new_audio_path, sr=32000, mono=True)
                audio = audio[None, :]  # (batch_size, segment_samples)

                print('------ Sound event detection ------')
                sed = SoundEventDetection(model = model, checkpoint_path=cp_path, device='cuda')
                framewise_output = sed.inference(audio)

                try:
                    os.makedirs(model_sound_name)
                except FileExistsError:
                    pass
                plot_sound_event_detection_result(framewise_output[0], model_sound_name, f"{p.split('.')[0]}.png")


if __name__ == '__main__':
    excluded_classes = ['AttBlock', 'ConvBlock', 'ConvBlock5x5',
                        'ConPreWavBlock', 'InvertedResidual', 'LogmelFilterBank',
                        'SpecAugmentation', 'Spectrogram', 'DaiNetResBlock',
                        'ConvPreWavBlock','LeeNetConvBlock', 'LeeNetConvBlock2', 'Wavegram_Logmel_Cnn14']

    # for files in os.listdir(PANN_PATH):
    #     try:
    #         files_part_find = files.index('_mAP')
    #         files_ext_find = files.index('.pth')
    #
    #         new_name = files[:files_part_find] + files[files_ext_find:]
    #         os.rename(os.path.join(PANN_PATH, files), os.path.join(PANN_PATH, new_name))
    #     except ValueError:
    #         pass
    #

    for index, m in enumerate(inspect.getmembers(models, inspect.isclass)):
        if m[0] not in excluded_classes:
            print(f"testing on model {m[0]}")
            try:
                try:
                    test_audio(m[0])
                except KeyError:
                    pass
                try:
                    test_sound(m[0])
                except KeyError:
                    pass
            except AssertionError:
                try:
                    test_audio(m[0], sample_rate_folder='16k', sample_rate=16000, window_size = 512, hop_size = 160, fmax = 8000)
                except AssertionError:
                    test_audio(m[0], sample_rate_folder='8k', sample_rate= 8000, window_size = 256, hop_size = 80, fmax = 4000)
