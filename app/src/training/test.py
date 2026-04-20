
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.config import Configuration
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from maikol_utils.print_utils import print_color

def test_model(CONFIG: Configuration, survival_module: torch.nn.Module, test_loader: DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    survival_module = survival_module.to(device)
    survival_module.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            try:
                image = batch['image'].to(device)
                tabular = batch['tabular'].to(device)
                labels = batch['label'].to(device)
            except (TypeError, KeyError):
                image, tabular, labels = batch
                image = image.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)
            preds = survival_module(image, tabular)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    print_color(f" - Test F1 Score: {f1:.4f}", 'green')
    print_color(f" - Test Precision: {precision:.4f}", 'green')
    print_color(f" - Test Recall: {recall:.4f}", 'green')
    print_color(f" - Test Accuracy: {accuracy:.4f}", 'green')

    return {
        'predictions': all_preds,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


