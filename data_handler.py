import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ASVSpoof5Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading ASVSpoof 5 audio files [cite: 8, 29].
    This class handles the ingestion of raw waveforms specifically for
    the neural denoising U-Net front-end.
    """
    def __init__(self, metadata_path, audio_dir):
        """
        Initializes the dataset by loading the metadata and setting the audio directory.
        :param metadata_path: Path to the TSV file containing ASVSpoof 5 labels, attack type and codec details.
        :param audio_dir: Directory containing the raw ASVSpoof 5 FLAC audio files.
        """
        # Read TSV metadata file containing labels, attach type and codec details
        self.metadata = pd.read_csv(
            metadata_path, sep=' ', header=None,
            names=['SPEAKER_ID', 'FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC',
                   'CODEC_Q', 'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
        )
        self.audio_dir = audio_dir

    def __len__(self):
        """
        Returns the total number of utterances in this dataset split.
        :return: Am integer representing the total number of audio files in the dataset metadata.
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Fetches and loads a single raw audio waveform and its corresponding binary label.
        :param idx: The integer index of the item to retrieve from the metadata dataframe.
        :return: A tuple containing the 1D raw audio waveform tensor and its corresponding binary label tensor.
        """
        row = self.metadata.iloc[idx]

        # Construct the file path to the .flac audio
        file_path = os.path.join(self.audio_dir, row['FLAC_FILE_NAME'] + '.flac')

        # Binary classification:
        # 1.0 for human speech and 0.0 for fake
        label = 1.0 if row['KEY'] == 'bonafide' else 0.0
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # Load raw audio
        waveform, sample_rate = torchaudio.load(file_path)

        # Squeeze out the channel dimension so it's a flat 1D array [time]
        waveform = waveform.squeeze(0)

        return waveform, label_tensor


def pad_collate_fn(batch):
    """
    Pads 1D raw waveforms so they can be processed in uniform batches by the dataloader.
    This allows the dataloader to handle variable lengths sequence without truncating temporal data.
    :param batch: A list of tuples, where each tuple contains (waveform, label_tensor) returned by __getitem__.
    :return: A tuple of three tensors: padded waveforms, stacked labels and original sequence lengths.
    """
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad raw waves with digital silence (0.0) to match longest sequence in batch
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels_tensor = torch.stack(labels)

    # Save original lengths to properly pack the sequence later for BiLSTM
    # Prevents RNN from calculating hidden states on the empty padding
    lengths = torch.tensor([len(w) for w in waveforms])


    return padded_waveforms, labels_tensor, lengths


if __name__ == "__main__":
    # Paths to ASVspoof 5
    TRAIN_META_PATH = "ASVspoof5/ASVspoof5.train.tsv"
    TRAIN_AUDIO_DIR = "ASVspoof5/flac_T"

    # Initialize dataset
    print("Loading metadata...")
    dataset = ASVSpoof5Dataset(metadata_path=TRAIN_META_PATH, audio_dir=TRAIN_AUDIO_DIR)
    print(f"Total files in dataset: {len(dataset)}")

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)

    # Fetch one batch to verify shapes and padding logic
    print("Fetching one batch of audio...")
    for features, labels, lengths in dataloader:
        print("--- Batch Info ---")
        print(f"Features shape (Batch Size, Time Steps, Feature): {features.shape}")
        print(f"Labels (1=Real, 0=Fake): {labels})")
        print(f"Original sequence lengths: {lengths}")
        break
