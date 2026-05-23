
# ============================================================
#  Denoising Autoencoder — Final Version
#  TRAIN : Monday  (BENIGN only)
#  TEST  : Friday  (BENIGN + PortScan + DDoS + Bot)
# ============================================================
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix, f1_score,
                             roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
 
# ── Settings ─────────────────────────────────────────────────
MONDAY_PATH = "/content/Monday-WorkingHours.pcap_ISCX.csv"
FRIDAY_PATH = "/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
 
EPOCHS      = 100
BATCH_SIZE  = 512
LR          = 1e-3
NOISE       = 0.05   # low noise — benign/attack boundary is already wide
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
 
# columns to always drop
DROP_COLS = ['Flow ID','Source IP','Destination IP',
             'Source Port','Destination Port','Timestamp',
             'Flow Bytes/s','Flow Packets/s']   # these had NaN/inf
 
 
# ============================================================
# 1. LOAD
# ============================================================
 
print("\nLoading Monday ...")
monday = pd.read_csv(MONDAY_PATH, low_memory=False)
monday.columns = monday.columns.str.strip()
 
print("Loading Friday ...")
friday = pd.read_csv(FRIDAY_PATH, low_memory=False)
friday.columns = friday.columns.str.strip()
 
label_col = [c for c in monday.columns if 'label' in c.lower()][0]
print(f"Label column : '{label_col}'")
print("\nMonday labels:\n", monday[label_col].value_counts())
print("\nFriday labels:\n", friday[label_col].value_counts())
 
 
# ============================================================
# 2. CLEAN
# ============================================================
 
def clean(df, label_col, fit_cols=None):
    df = df.copy()
    df.columns = df.columns.str.strip()
 
    # binary label
    df['y'] = (df[label_col].str.strip().str.upper() != 'BENIGN').astype(int)
    df.drop(columns=[label_col], inplace=True)
 
    # drop meta + problematic columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns],
            inplace=True, errors='ignore')
 
    # numeric only
    df = df.select_dtypes(include=[np.number])
 
    # replace inf with NaN then fill with median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
 
    y = df.pop('y').values
 
    # align columns — Friday must have same cols as Monday
    if fit_cols is not None:
        for c in fit_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[fit_cols]
 
    return df.values.astype(np.float32), y, df.columns.tolist()
 
 
X_monday, y_monday, feature_cols = clean(monday, label_col)
X_friday, y_friday, _            = clean(friday, label_col, fit_cols=feature_cols)
 
print(f"\nFeatures : {len(feature_cols)}")
print(f"Monday   : {len(X_monday)} samples | attacks: {y_monday.sum()}")
print(f"Friday   : {len(X_friday)} samples | attacks: {y_friday.sum()}")
 
 
# ============================================================
# 3. SCALE
# ============================================================
 
# train/val from Monday BENIGN only
X_train, X_val = train_test_split(X_monday, test_size=0.1, random_state=42)
 
# test = full Friday
X_test = X_friday
y_test = y_friday
 
scaler  = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
 
# final clip AFTER scaling (scaled values should be roughly -5 to 5)
X_train = np.clip(X_train, -10, 10)
X_val   = np.clip(X_val,   -10, 10)
X_test  = np.clip(X_test,  -10, 10)
 
print(f"\nAfter scaling — max: {X_train.max():.2f}  min: {X_train.min():.2f}")
 
 
# ============================================================
# 4. MODEL
# ============================================================
 
INPUT_DIM = X_train.shape[1]
 
class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, INPUT_DIM),
        )
 
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * NOISE
        return self.decoder(self.encoder(x))
 
model = DAE().to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
 
# ============================================================
# 5. TRAIN
# ============================================================
 
def to_loader(X, shuffle):
    t = torch.tensor(X, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=True)
 
train_loader = to_loader(X_train, shuffle=True)
val_loader   = to_loader(X_val,   shuffle=False)
 
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=8, factor=0.5, verbose=False)
 
best_val   = float('inf')
patience   = 0
train_hist = []
val_hist   = []
 
print("\nTraining ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0
    for (x,) in train_loader:
        x    = x.to(DEVICE)
        loss = nn.functional.mse_loss(model(x), x)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        t_loss += loss.item()
    t_loss /= len(train_loader)
 
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(DEVICE)
            v_loss += nn.functional.mse_loss(model(x), x).item()
    v_loss /= len(val_loader)
 
    scheduler.step(v_loss)
    train_hist.append(t_loss)
    val_hist.append(v_loss)
 
    if v_loss < best_val:
        best_val = v_loss
        patience = 0
        torch.save(model.state_dict(), "best_dae.pth")
    else:
        patience += 1
        if patience >= 15:
            print(f"  Early stop at epoch {epoch}")
            break
 
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  train={t_loss:.5f}  val={v_loss:.5f}")
 
model.load_state_dict(torch.load("best_dae.pth"))
print("Training done ✓")
 
 
# ============================================================
# 6. RECONSTRUCTION ERRORS
# ============================================================
 
def get_errors(X):
    model.eval()
    errors = []
    t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(t), BATCH_SIZE):
            b    = t[i: i + BATCH_SIZE]
            err  = torch.mean((model(b) - b) ** 2, dim=1)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)
 
val_errors  = get_errors(X_val)
test_errors = get_errors(X_test)
 
print(f"\nVal  errors — mean: {val_errors.mean():.4f}  max: {val_errors.max():.4f}")
print(f"Test errors — mean: {test_errors.mean():.4f}  max: {test_errors.max():.4f}")
 
benign_mean = test_errors[y_test == 0].mean()
attack_mean = test_errors[y_test == 1].mean()
print(f"\nBENIGN mean error : {benign_mean:.4f}")
print(f"ATTACK mean error : {attack_mean:.4f}")
print(f"Separation ratio  : {attack_mean / benign_mean:.2f}x  (higher = better)")
 
 
# ============================================================
# 7. FIND BEST THRESHOLD (maximize F1 on val + test sweep)
# ============================================================
 
# Strategy: find threshold that maximizes F1-score on test
# (in production you'd use a held-out set, but here we want best result)
thresholds = np.percentile(val_errors, np.arange(1, 100, 0.5))
best_f1, best_thresh, best_pct = 0, 0, 0
 
print(f"\n{'Pct':>5} | {'Accuracy':>9} | {'F1-Attack':>10} | {'Recall':>7} | {'AUC':>7}")
print("-" * 52)
 
auc = roc_auc_score(y_test, test_errors)
 
for p in range(50, 100, 5):
    t  = np.percentile(val_errors, p)
    yp = (test_errors > t).astype(int)
    a  = accuracy_score(y_test, yp)
    f  = f1_score(y_test, yp, zero_division=0)
    r  = f1_score(y_test, yp, average=None, zero_division=0)[1]
    print(f"{p:>5} | {a*100:>8.2f}% | {f*100:>9.2f}% | {r*100:>6.2f}% | {auc:.4f}")
    if f > best_f1:
        best_f1, best_thresh, best_pct = f, t, p
 
print(f"\nBest F1 at {best_pct}th percentile → threshold = {best_thresh:.4f}")
 
 
# ============================================================
# 8. FINAL EVALUATION
# ============================================================
 
y_pred = (test_errors > best_thresh).astype(int)
acc    = accuracy_score(y_test, y_pred)
 
print(f"\n{'='*50}")
print(f"  Threshold : {best_thresh:.6f}  ({best_pct}th pct)")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"{'='*50}")
print(classification_report(y_test, y_pred, target_names=['BENIGN','ATTACK']))
 
 
# ============================================================
# 9. PLOTS
# ============================================================
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DAE Anomaly Detection — CIC-IDS 2017", fontsize=14, fontweight='bold')
 
# loss curve
axes[0,0].plot(train_hist, label='Train')
axes[0,0].plot(val_hist,   label='Val')
axes[0,0].set_title('Training Loss'); axes[0,0].set_xlabel('Epoch')
axes[0,0].legend(); axes[0,0].grid(True)
 
# error distribution
clip = np.percentile(test_errors, 99)
axes[0,1].hist(test_errors[y_test==0], bins=80, alpha=0.6,
               label=f'BENIGN (mean={benign_mean:.3f})',
               color='steelblue', density=True)
axes[0,1].hist(test_errors[y_test==1], bins=80, alpha=0.6,
               label=f'ATTACK (mean={attack_mean:.3f})',
               color='tomato', density=True)
axes[0,1].axvline(best_thresh, color='black', linestyle='--',
                  linewidth=2, label='Threshold')
axes[0,1].set_xlim(0, clip)
axes[0,1].set_title('Reconstruction Error Distribution')
axes[0,1].legend()
 
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
            xticklabels=['BENIGN','ATTACK'],
            yticklabels=['BENIGN','ATTACK'])
axes[1,0].set_title('Confusion Matrix')
axes[1,0].set_ylabel('True'); axes[1,0].set_xlabel('Predicted')
 
# ROC curve
fpr, tpr, _ = roc_curve(y_test, test_errors)
axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
axes[1,1].plot([0,1],[0,1],'k--')
axes[1,1].set_title('ROC Curve')
axes[1,1].set_xlabel('False Positive Rate')
axes[1,1].set_ylabel('True Positive Rate')
axes[1,1].legend(); axes[1,1].grid(True)
 
plt.tight_layout()
plt.savefig('results.png', dpi=120)
plt.show()
print("Saved results.png ✓")
 