import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data_handler import ASVSpoof5Dataset, pad_collate_fn
from crnn_model import DeepfakeCRNN
from evaluate import compute_eer


def calculate_accuracy(predictions, labels):
    """
    Calculates the binary classification accuracy for a single batch.
    While EER is our primary metric, tracking accuracy helps ensure the BCE loss is converging.
    :param predictions: A 1D tensor of predicted probabilities (0.0 to 1.0) outputted by the model.
    :param labels: A 1D tensor of ground truth binary labels (1.0 for bonafide, 0.0 for spoof).
    :return: A float representing the percentage of correct predictions in the batch.
    """
    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).float()
    return (correct.sum() / len(correct)).item()


def train_master_pipeline():
    """
    Executes Stage 2 Detection training for the Vox Veritas CRNN pipeline[cite: 1].
    This function loads the pre-trained U-Net, freezes its weights to act purely as an
    enhancement front-end , and trains the downstream CNN and BiLSTM on the ASVspoof 5 dataset.
    :return: None
    """
    # Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Master CRNN Pipeline on: {device}")

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    EPOCHS = 10

    # Initialize Model and Inject the Pre-trained Denoiser
    model = DeepfakeCRNN(rnn_hidden_size=64).to(device)

    print("\nLoading Pre-trained U-Net Denoiser...")
    try:
        model.denoiser.load_state_dict(torch.load("pretrained_unet.pth", weights_only=True))

        # Freeze U-Net so it acts purely as an enhancement front-end
        # Only train the CNN and RNN on the deepfake artifacts
        for param in model.denoiser.parameters():
            param.requires_grad = False
        print("Denoiser loaded and frozen. It will act purely as an enhancement front-end.")
    except FileNotFoundError:
        print("Warning: 'pretrained_unet.pth' not found. Did you run pretrain_denoiser.py?")
        return

    # Load ASVspoof 5 Dataset
    print("\nLoading ASVspoof 5 Datasets...")

    # Train set
    # Used for updating model weights
    train_dataset = ASVSpoof5Dataset(
        metadata_path="ASVspoof5/ASVspoof5.train.tsv",
        audio_dir="ASVspoof5/flac_T"
    )

    # Dev set
    # Used for validation to test against unseen speakers
    val_dataset = ASVSpoof5Dataset(
        metadata_path="ASVspoof5/ASVspoof5.dev.tsv",
        audio_dir="ASVspoof5/flac_D"
    )

    print(f"Training Files: {len(train_dataset)}")
    print(f"Validation Files: {len(val_dataset)}")

    # num_workers increased to 4 to feed GPU
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=pad_collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=pad_collate_fn, num_workers=4
    )

    # Loss and Optimizer
    # Binary Cross-Entropy loss used for detection stage
    criterion = nn.BCELoss()
    # Only pass parameters require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # Track best EER
    best_val_eer = float('inf')

    # Training and validation loop
    print(f"\nStarting CRNN Training for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        model.denoiser.eval()

        train_loss, train_acc = 0.0, 0.0

        for features, labels, lengths in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            predictions = model(features, lengths)
            loss = criterion(predictions, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(predictions, labels)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0

        # Accumulate all predictions and labels for EER calculation
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features, labels = features.to(device), labels.to(device)

                predictions = model(features, lengths)

                loss = criterion(predictions, labels)
                val_loss += loss.item()
                val_acc += calculate_accuracy(predictions, labels)

                # Append batch arrays to our global lists
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        # Calculate Epoch EER across entire validation set
        val_eer, val_threshold = compute_eer(np.array(all_val_labels), np.array(all_val_preds))

        # Metric output
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.4f} | Val EER: {val_eer * 100:.2f}%")

        # Save model when the EER improves
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), "best_master_crnn.pth")
            print(f"Validation EER improved to {val_eer * 100:.2f}%! Master CRNN saved.")


if __name__ == "__main__":
    train_master_pipeline()
