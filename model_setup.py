from config import classes_num
import models

SAMPLE_RATE = 32000
WINDOWS_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
CLASSES_NUM = classes_num


def set_model(model_name, sample_rate = SAMPLE_RATE, window_size = WINDOWS_SIZE,
              hop_size = HOP_SIZE, mel_bins = MEL_BINS, fmin = FMIN, fmax = FMAX):
    model = getattr(models, model_name)
    return model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, CLASSES_NUM)