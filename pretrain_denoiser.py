import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader

from crnn_model import UNetDenoiser


class NoisyCleanDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paired clean and noisy audio.
    This supports Stage 1 Enhancement, where the lightweight U-Net front-end is
    pre-trained on an auxiliary noisy speech corpus to learn robust speech separation
    before being frozen for the deepfake detection pipeline.
    """
    def __init__(self, clean_dir, noisy_dir, fixed_length=32000):
        """
        Initializes the dataset by setting the directories for clean and noisy audio pairs.
        :param clean_dir: Folder containing the perfect studio-quality clean audio files.
        :param noisy_dir: Folder containing the exact same audio files, but with added noise/distortion.
        :param fixed_length: The fixed number of samples the U-Net expects (default 32000 samples = 2 seconds at 16kHz).
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.fixed_length = fixed_length
        self.file_names = os.listdir(clean_dir)

    def __len__(self):
        """
        Returns the total number of audio file pairs in this dataset.
        :return: An integer representing the total number of files in the clean directory.
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Fetches and loads a single pair of clean and noisy audio waveforms, padded or trimmed to a fixed length.
        :param idx: The integer index of the file to retrieve from the directory list.
        :return: A tuple containing two 2D tensors: the noisy waveform [channels, time] and the clean waveform [channels, time].
        """
        file_name = self.file_names[idx]
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)

        # Load audio pairs
        clean_wave, _ = torchaudio.load(clean_path)
        noisy_wave, _ = torchaudio.load(noisy_path)

        # Force audio to be exactly 'fixed_length'
        # If it's too short pad with zeros
        # If it's too long cut it
        if clean_wave.shape[1] < self.fixed_length:
            pad_amount = self.fixed_length - clean_wave.shape[1]
            clean_wave = torch.nn.functional.pad(clean_wave, (0, pad_amount))
            noisy_wave = torch.nn.functional.pad(noisy_wave, (0, pad_amount))
        elif clean_wave.shape[1] > self.fixed_length:
            clean_wave = clean_wave[:, :self.fixed_length]
            noisy_wave = noisy_wave[:, :self.fixed_length]

        # The U-Net expects [channels, time]
        return noisy_wave, clean_wave


def pretrain_unet():
    """
    Executes Stage 1 Enhancement by training the U-Net denoiser to suppress background
    noise and mitigate heavy codec compression.
    """
    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pre-training Denoiser on Device: {device}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # Load Real VoiceBank-DEMAND Dataset
    print("\nLoading VoiceBank-DEMAND Dataset...")

    # Path to data
    CLEAN_DIR = "archive/clean_trainset_28spk_wav"
    NOISY_DIR = "archive/noisy_trainset_28spk_wav"

    dataset = NoisyCleanDataset(clean_dir=CLEAN_DIR, noisy_dir=NOISY_DIR, fixed_length=32000)

    # DataLoader configuration
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # Initialize Model and L1 Loss
    model = UNetDenoiser().to(device)

    # L1 Loss measures absolute distance between predicted wave and clean wave
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"\nStarting U-Net Pre-Training for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for noisy_batch, clean_batch in dataloader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()

            # Pass noisy audio through U-Net
            cleaned_audio = model(noisy_batch)

            # U-Net returns [batch, time]
            # Need to unsqueeze to [batch, 1, time]
            cleaned_audio = cleaned_audio.unsqueeze(1)

            # Calculate loss
            loss = criterion(cleaned_audio, clean_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Waveform L1 Loss: {avg_loss:.4f}")

    # Save trained Denoiser weights to be frozen later in the main CRNN pipeline
    torch.save(model.state_dict(), "pretrained_unet.pth")
    print("\nPre-training Complete! Saved 'pretrained_unet.pth'")


if __name__ == "__main__":
    pretrain_unet()
