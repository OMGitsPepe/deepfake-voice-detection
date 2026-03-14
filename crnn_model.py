import torch
import torch.nn as nn
import torchaudio


# Denoising Front-End
class UNetDenoiser(nn.Module):
    """
    Module A - Neural Denoising: A lightweight U-Net architecture applied to the raw waveform.
    This module suppresses background noise and mitigates severe codec distortion,
    making the downstream classifier focus strictly on vocal artifacts rather than
    environmental acoustic mismatches.
    """
    def __init__(self):
        """
        Initializes the encoder and decoder convolutional layers for the U-Net.
        """
        super(UNetDenoiser, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7)

        # Decoder (Upsampling)
        self.dec1 = nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec2 = nn.Conv1d(32, 1, kernel_size=15, stride=1, padding=7)  # 32 because of skip connection

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs the forward pass of the U-Net denoiser to enhance the raw audio signal.
        :param x: A 3D tensor of shape [batch_size, 1, time] representing the noisy input waveform.
        :return: A 2D tensor of shape [batch_size, time] representing the enhanced, denoised waveform.
        """
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))

        d1 = self.relu(self.dec1(e2))

        # If the upsampled tensor (d1) is slightly longer than the skip tensor (e1)
        # Trim the extra padding off end of d1
        if d1.size(2) != e1.size(2):
            d1 = d1[:, :, :e1.size(2)]

        # Skip connection concatenate encoder 1 output with decoder 1 output
        skip = torch.cat([d1, e1], dim=1)

        out = self.dec2(skip)

        # Return [batch, time]
        return out.squeeze(1)


# Spatial Feature Extractor (CNN)
class CNNExtractor(nn.Module):
    """
    Module B - Spatial Feature Extractor: A truncated ResNet/CNN that extracts
    frame-level spatial features from Mel-Spectrograms.
    """
    def __init__(self):
        """
        Initializes the 2D Convolutional sequence, batch normalization, and pooling layers.
        """
        super(CNNExtractor, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )

    def forward(self, x):
        """
        Processes the Mel-Spectrogram to capture local spectral anomalies without pooling the time dimension.
        :param x: A 4D tensor of shape [batch_size, channels, freq, time] representing the Mel-spectrogram.
        :return: A 3D tensor of shape [batch_size, time, features] representing the sequence of feature maps.
        """
        out = self.conv_blocks(x)
        out = out.squeeze(2).transpose(1, 2)
        return out


# CRNN Pipeline
class DeepfakeCRNN(nn.Module):
    """
    Module C - Temporal Sequence Modeling (BiLSTM) & Full Pipeline Integration.
    By processing the sequence both forward and backward, the BiLSTM captures long-range
    phonetic dependencies, temporal rhythm, and pacing anomalies.
    """
    def __init__(self, rnn_hidden_size=64):
        """
        Initializes the complete CRNN pipeline: Denoiser, Mel-transform, CNN, and BiLSTM.
        :param rnn_hidden_size: Integer defining the number of features in the BiLSTM hidden state.
        """
        super(DeepfakeCRNN, self).__init__()

        self.denoiser = UNetDenoiser()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64, n_fft=512, hop_length=160
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.cnn = CNNExtractor()

        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, 1)

    def forward(self, waveform, lengths):
        """
        Maps noisy acoustic waveforms to temporal deepfake probabilities.
        :param waveform: A 2D tensor of shape [batch_size, time] representing the raw padded audio sequences.
        :param lengths: A 1D tensor of original sequence lengths on the CPU (used for proper RNN packing).
        :return: A 1D tensor of shape [batch_size] containing the final deepfake probabilities (0.0 to 1.0).
        """
        # Denoise raw waveform
        waveform = waveform.unsqueeze(1)
        clean_waveform = self.denoiser(waveform)

        # Extract spatial features
        mel = self.melspec(clean_waveform)
        mel_db = self.amplitude_to_db(mel)
        mel_db = mel_db.unsqueeze(1)

        cnn_features = self.cnn(mel_db)

        # Calculate sequence lengths after MelSpectrogram hop-length reduction
        cnn_lengths = (lengths // 160) + 1

        # Pack sequence to skip padded zeros during RNN computation
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            cnn_features, cnn_lengths, batch_first=True, enforce_sorted=False
        )

        # Temporal modeling via BiLSTM
        _, (hidden, _) = self.rnn(packed_input)

        # Concatenate final forward and backward hidden states
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Final classification
        logits = self.fc(final_hidden)
        return torch.sigmoid(logits).squeeze(1)


if __name__ == "__main__":
    mock_waveform = torch.randn(4, 16000)
    mock_lengths = torch.tensor([16000, 15000, 14000, 16000])

    model = DeepfakeCRNN()
    predictions = model(mock_waveform, mock_lengths)

    print("CRNN Model Test:")
    print(f"Output Shape: {predictions.shape}")
    print(f"Probabilities: {predictions}")
