import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# ---------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------
# 1. Evidential Loss Function (Subjective Logic)
# ---------------------------------------------------------
class EvidentialMSELoss(nn.Module):
    def __init__(self, num_classes=4, anneal_epochs=10):
        super(EvidentialMSELoss, self).__init__()
        self.num_classes = num_classes
        self.anneal_epochs = anneal_epochs

    def kl_divergence(self, alpha):
        beta = torch.ones((1, self.num_classes), dtype=torch.float32, device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_prior = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dgAlpha = torch.digamma(alpha)
        dgSAlpha = torch.digamma(S_alpha)
        
        kl = torch.sum((alpha - beta) * (dgAlpha - dgSAlpha), dim=1, keepdim=True) + lnB + lnB_prior
        return kl

    def forward(self, logits, target, epoch):
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=1, keepdim=True)
        p_hat = alpha / S
        
        loss_mse = torch.sum((y_one_hot - p_hat) ** 2, dim=1, keepdim=True)
        loss_var = torch.sum(alpha * (S - alpha) / (S ** 2 * (S + 1.0)), dim=1, keepdim=True)
        loss_evidential = torch.mean(loss_mse + loss_var)
        
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        kl_loss = torch.mean(self.kl_divergence(alpha_tilde))
        
        anneal_coef = min(1.0, float(epoch) / self.anneal_epochs)
        return loss_evidential + anneal_coef * kl_loss

# ---------------------------------------------------------
# 2. Evidential Classifier Architecture
# ---------------------------------------------------------
class EvidentialALLClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(EvidentialALLClassifier, self).__init__()
        self.num_classes = num_classes
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x):
        self.eval()
        logits = self.forward(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        probs = alpha / S
        uncertainty = self.num_classes / S
        return probs, uncertainty

# ---------------------------------------------------------
# 3. Stratified Data Pipeline
# ---------------------------------------------------------
def get_stratified_dataloaders(data_dir, batch_size=32, image_size=224):
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir)
    targets = dataset.targets

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), 
        test_size=0.2, 
        random_state=42, 
        stratify=targets
    )

    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=transform_train), train_idx)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=transform_val), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, dataset.classes

# ---------------------------------------------------------
# 4. Publication Style & Visualization Modules
# ---------------------------------------------------------
def set_publication_style():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf"
    })

def generate_uncertainty_density_pdf(uncertainties, labels, preds, save_path="uncertainty_density.pdf"):
    set_publication_style()
    correct_u = np.array(uncertainties)[np.array(preds) == np.array(labels)]
    incorrect_u = np.array(uncertainties)[np.array(preds) != np.array(labels)]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(correct_u, fill=True, label="Correct Predictions", color="blue", ax=ax)
    sns.kdeplot(incorrect_u, fill=True, label="Incorrect Predictions", color="red", ax=ax)
    
    ax.set_xlabel("Epistemic Uncertainty (u = K / S)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution Analysis")
    ax.legend(loc="upper right")
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved publication figure: {save_path}")

def generate_confusion_matrix_pdf(labels, preds, class_names, save_path="confusion_matrix.pdf"):
    set_publication_style()
    cm = __import__('sklearn.metrics').metrics.confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Classification Performance")
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved publication figure: {save_path}")

def generate_roc_curves_pdf(labels, probs, class_names, save_path="roc_curves.pdf"):
    set_publication_style()
    labels = np.array(labels)
    probs = np.array(probs)
    num_classes = len(class_names)
    labels_one_hot = np.eye(num_classes)[labels]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['blue', 'red', 'green', 'purple']
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curves")
    ax.legend(loc="lower right")
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved publication figure: {save_path}")

def generate_reliability_diagram_pdf(labels, preds, probs, save_path="reliability_diagram.pdf"):
    set_publication_style()
    confidences = np.max(np.array(probs), axis=1)
    accuracies = (np.array(preds) == np.array(labels))
    
    num_bins = 10
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=True)
    
    bin_accs, bin_confs = [], []
    for b in range(1, num_bins + 1):
        mask = (bin_indices == b)
        if np.any(mask):
            bin_accs.append(np.mean(accuracies[mask]))
            bin_confs.append(np.mean(confidences[mask]))
            
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    ax.plot(bin_confs, bin_accs, "s-", color="blue", label="Evidential Model")
    
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left")
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved publication figure: {save_path}")

def generate_confidence_vs_uncertainty_pdf(probs, uncertainties, labels, preds, save_path="conf_vs_uncertainty.pdf"):
    set_publication_style()
    confidences = np.max(np.array(probs), axis=1)
    uncertainties = np.array(uncertainties)
    correct_mask = (np.array(preds) == np.array(labels))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(confidences[~correct_mask], uncertainties[~correct_mask], color='red', alpha=0.7, edgecolors='k', label="Incorrect Predictions")
    ax.scatter(confidences[correct_mask], uncertainties[correct_mask], color='blue', alpha=0.3, edgecolors='none', label="Correct Predictions")
    
    ax.set_xlabel("Top-1 Predicted Confidence")
    ax.set_ylabel("Epistemic Uncertainty (u = K / S)")
    ax.set_title("Confidence vs. Epistemic Uncertainty")
    ax.legend(loc="upper right")
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved publication figure: {save_path}")

# ---------------------------------------------------------
# 5. Metrics Calculation
# ---------------------------------------------------------
def calculate_comprehensive_metrics(labels, preds, probs):
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(labels, preds, digits=4))
    
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    brier_score = np.mean(np.sum((probs - labels_one_hot)**2, axis=1))
    print(f"Multi-class Brier Score: {brier_score:.4f} (Lower is better)")
    
    confidences = np.max(probs, axis=1)
    accuracies = (preds == labels)
    
    num_bins = 10
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=True)
    
    ece = 0.0
    for b in range(1, num_bins + 1):
        mask = (bin_indices == b)
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_weight = np.sum(mask) / len(confidences)
            ece += np.abs(bin_acc - bin_conf) * bin_weight
            
    print(f"Expected Calibration Error (ECE): {ece:.4f} (Lower is better)\n")
    return brier_score, ece

# ---------------------------------------------------------
# 6. Training Loop
# ---------------------------------------------------------
def train_evidential_model(model, train_loader, val_loader, epochs=20, device="cuda"):
    model.to(device)
    criterion = EvidentialMSELoss(num_classes=model.num_classes, anneal_epochs=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels, epoch=epoch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        correct, total = 0, 0
        total_u = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                probs, uncertainty = model.predict_with_uncertainty(images)
                
                preds = torch.argmax(probs, dim=1)
                correct += (preds == labels).sum().item()
                total_u += uncertainty.sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_u = total_u / total
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val Uncertainty (u): {avg_u:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_evidential_model.pth")

    return model

# ---------------------------------------------------------
# 7. External Validation Loop
# ---------------------------------------------------------
def evaluate_external_dataset(model_path, data_dir, primary_classes, num_classes=4, device="cuda"):
    set_seed(42)
    model = EvidentialALLClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform_test)
    
    new_class_to_idx = {}
    for primary_class in primary_classes:
        for ext_folder in test_dataset.classes:
            if primary_class.lower() in ext_folder.lower():
                new_class_to_idx[ext_folder] = primary_classes.index(primary_class)
                break
                
    old_idx_to_new_idx = {
        i: new_class_to_idx[folder_name] 
        for i, folder_name in enumerate(test_dataset.classes)
    }
    
    test_dataset.class_to_idx = new_class_to_idx
    test_dataset.samples = [(path, old_idx_to_new_idx[old_idx]) for path, old_idx in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    
    all_preds, all_labels, all_uncertainties, all_probs = [], [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs, uncertainty = model.predict_with_uncertainty(images)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_u = np.mean(all_uncertainties)
    
    print(f"\n--- External Validation Summary ---")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Average Uncertainty: {avg_u:.4f}")
    
    calculate_comprehensive_metrics(all_labels, all_preds, all_probs)
    
    generate_confusion_matrix_pdf(all_labels, all_preds, primary_classes)
    generate_uncertainty_density_pdf(all_uncertainties, all_labels, all_preds)
    generate_roc_curves_pdf(all_labels, all_probs, primary_classes)
    generate_reliability_diagram_pdf(all_labels, all_preds, all_probs)
    generate_confidence_vs_uncertainty_pdf(all_probs, all_uncertainties, all_labels, all_preds)
    
    return all_preds, all_labels, all_uncertainties, all_probs

# ---------------------------------------------------------
# 8. Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    PRIMARY_DATA_DIR = "/kaggle/input/datasets/mehradaria/leukemia/Original"
    EXTERNAL_DATA_DIR = "/kaggle/input/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]"
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading primary dataset...")
    train_loader, val_loader, class_names = get_stratified_dataloaders(PRIMARY_DATA_DIR, batch_size=32)
    print(f"Detected classes: {class_names}")
    
    print("Initializing Evidential Classifier...")
    model = EvidentialALLClassifier(num_classes=4, pretrained=True)
    
    print("Starting training phase...")
    trained_model = train_evidential_model(model, train_loader, val_loader, epochs=20, device=device)
    
    print("Starting external validation and figure generation...")
    evaluate_external_dataset(
        model_path="best_evidential_model.pth", 
        data_dir=EXTERNAL_DATA_DIR, 
        primary_classes=class_names, 
        num_classes=4, 
        device=device
    )
    print("Pipeline execution complete.")
