import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize accumulators
all_y_true = []
all_y_pred = []

# Load truth/prediction files
truth_files = sorted(glob.glob("groundtruth1.csv"))
pred_files = sorted(glob.glob("final.csv"))

# Check for matching files
if len(truth_files) != len(pred_files):
    raise ValueError("Mismatch in number of truth and prediction files.")

# Collect data
for truth_file, pred_file in zip(truth_files, pred_files):
    df_truth = pd.read_csv(truth_file)
    df_pred = pd.read_csv(pred_file)

    df_merged = pd.merge(df_truth, df_pred, on="frame_id", suffixes=("_true", "_pred"))
    df_merged.dropna(subset=["hand_sign_true", "hand_sign_pred"], inplace=True)

    all_y_true.extend(df_merged['hand_sign_true'].astype(str).values)
    all_y_pred.extend(df_merged['hand_sign_pred'].astype(str).values)

# Encode string labels to integers
label_encoder = LabelEncoder()
all_y_true_enc = label_encoder.fit_transform(all_y_true)
all_y_pred_enc = label_encoder.transform(all_y_pred)
class_names = label_encoder.classes_
n_classes = len(class_names)

# Simulate softmax-style probability predictions (for illustration only)
y_pred_probs = np.zeros((len(all_y_pred_enc), n_classes))
for i, label in enumerate(all_y_pred_enc):
    y_pred_probs[i, label] = 1.0

# Compute metrics
accuracy = accuracy_score(all_y_true_enc, all_y_pred_enc)
recall = recall_score(all_y_true_enc, all_y_pred_enc, average='macro')
f1 = f1_score(all_y_true_enc, all_y_pred_enc, average='macro')
cm = confusion_matrix(all_y_true_enc, all_y_pred_enc)

# Specificity calculation
specificity_per_class = []
for i in range(n_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp)
    specificity_per_class.append(specificity)
specificity = np.mean(specificity_per_class)

# Binarize true labels
y_true_bin = label_binarize(all_y_true_enc, classes=list(range(n_classes)))


# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall - Macro Avg): {recall:.4f}")
print(f"Specificity (Average): {specificity:.4f}")
print(f"F1 Score (Macro Avg): {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(all_y_true_enc, all_y_pred_enc, target_names=class_names))



# === Confusion Matrix Plot ===
df_cmx = pd.DataFrame(cm, index=class_names, columns=class_names)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(df_cmx, annot=True, fmt='g', square=False,
            xticklabels=class_names, yticklabels=class_names,
            cmap='magma', annot_kws={"size": 12},
            cbar_kws={"shrink": 0.8})


ax.set_xticklabels(class_names, fontsize=12, rotation=45, fontstyle='normal')  # 
ax.set_yticklabels(class_names, fontsize=12, rotation=0, fontstyle='normal')   # 

ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title('Confusion Matrix (Video Evaluation)', fontsize=16)
ax.set_ylim(len(class_names), 0)

plt.tight_layout()
plt.show()

# ROC and AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curves
# plt.figure(figsize=(8, 6))
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
# plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve (Video Evaluation)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
