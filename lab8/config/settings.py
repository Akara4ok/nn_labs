#Data
DATA_PATH = 'Data/LJSpeech-1.1'
CHARLIST_PATH = 'Artifacts/utils/charlist.txt'
SPELL_FILE_PATH = 'Artifacts/utils/spell_correction.txt'

#Tensorflow
VAL_PERCENT = 0.1
TEST_PERCENT = 0.2
RANDOM_SEED = 42
SAVE_FOLDER = 'Artifacts/Models'
BATCH_SIZE = 32

#Model
EPOCHS = 5
DEFAULT_LR = 0.0001

#Spectrogram
FRAME_LENGTH = 256
FRAME_STEP = 160
FFT_LENGTH = 384