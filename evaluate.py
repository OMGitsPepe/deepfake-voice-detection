import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from crnn_model import DeepfakeCRNN
from data_handler import ASVSpoof5Dataset, pad_collate_fn


def compute_eer(labels, scores):
    """
    Calculates the Equal Error Rate (EER), the primary evaluation metric for the Vox Veritas project to achieve a
    development-set EER competitive with published SOTA on ASVspoof 5. EER is the point where the False
    Positive Rate mathematically equals the False Negative Rate.
    :param labels: A 1D numpy array or list of true binary labels (1.0 for bonafide Real, 0.0 for Fake).
    :param scores: A 1D numpy array or list of predicted probabilities outputted by the model.
    :return: A tuple containing the calculated EER (float) and the optimal probability threshold (float)
    to achieve this EER.
    """
    # Calculate False Positive Rate and True Positive Rate
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # EER is the point where FPR mathematically equals FNR
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # Find specific probability threshold
    threshold = interp1d(fpr, thresholds)(eer)

    return eer, threshold


def evaluate_model():
    """
    Executes the final evaluation phase of the Vox Veritas CRNN pipeline to measure performance on ASVspoof 5,
    which contains ~182K utterances spanning diverse TTS and VC attack types under real-world
    channel conditions[cite: 29].
    """
    # Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Evaluation on: {device}")

    # Load the Dataset
    print("Loading ASVspoof 5 Development Dataset for Evaluation...")

    # Using dev set to test on unseen data
    test_dataset = ASVSpoof5Dataset(
        metadata_path="ASVspoof5/ASVspoof5.eval.track_1.tsv",
        audio_dir="ASVspoof5/flac_E_eval"
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        collate_fn=pad_collate_fn, num_workers=0
    )

    print(f"Total Test Files: {len(test_dataset)}")

    # Load trained model
    model = DeepfakeCRNN(rnn_hidden_size=64).to(device)
    try:
        model.load_state_dict(torch.load("best_master_crnn.pth", weights_only=True))
        print("Successfully loaded 'best_master_crnn.pth'")
    except FileNotFoundError:
        print("Error: 'best_master_crnn.pth' not found. You need to train the model first!")
        return

    # Put model in evaluation mode
    model.eval()

    all_predictions = []
    all_labels = []

    # Run Inference
    print("\nRunning test data through the CRNN...")
    with torch.no_grad():
        for i, (features, labels, lengths) in enumerate(test_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Get probabilities scores from model
            predictions = model(features, lengths)

            # Move tensors to CPU and convert to standard Python/Numpy lists
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 20 == 0:
                print(f"Processed batch {i + 1}/{len(test_loader)}")

    # Calculate metrics
    print("\nCalculating Final Metrics...")
    eer, threshold = compute_eer(np.array(all_labels), np.array(all_predictions))

    print("=" * 40)
    print("FINAL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"Optimal Threshold:      {threshold:.4f}")
    print("=" * 40)
    print("*(If your probability is > Optimal Threshold, the model guesses Real)*")
    print("*(If your probability is < Optimal Threshold, the model guesses Fake)*")


if __name__ == "__main__":
    evaluate_model()
