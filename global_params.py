# Global parameters

BATCH_SIZE = 128
NUM_CLASSES = 62
IMG_SIZE = 28
PIXEL_THRESH = 0.30
MIN_AREA = 40
CHAR_MAPPING = {"J": "I",
                "L": "I",
                "1": "I",
                "T": "I",
                "Z": "I",
                "Q": "O",
                "A": "O",
                "0": "O",
                "D": "O",
                "V": "Y",
                "5": "S",
                "9": "G",
                "2": "Z",
                "C": "G",
                "8": "B"}