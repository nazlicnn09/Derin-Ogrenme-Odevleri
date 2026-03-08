import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. VERİ SETİ: XOR Kapısı
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2. MODEL MİMARİSİ: Çok Katmanlı Perceptron (MLP)
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()

        self.hidden = nn.Linear(2, 8)

        self.output = nn.Linear(8, 1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = XORNet()

# 3. KAYIP VE OPTİMİZASYON
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. EĞİTİM DÖNGÜSÜ
epochs = 5000
losses = []

print("Eğitim başlıyor...")
for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Kayıp (Loss): {loss.item():.4f}")

# 5. GÖRSELLEŞTİRME: Kayıp Grafiği
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Eğitim Kayıp (Loss) Grafiği")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

# 6. GÖRSELLEŞTİRME: Karar Sınırı
plt.subplot(1, 2, 2)
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    probs = model(grid).reshape(xx.shape)

plt.contourf(xx, yy, probs, cmap="RdBu", alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), edgecolors='k', cmap="RdBu")
plt.title("XOR Karar Sınırı (Decision Boundary)")

plt.tight_layout()
plt.show()

# Test Çıktısı
print("\nFinal Tahminleri:")
with torch.no_grad():
    outputs = model(X)
    for i in range(len(X)):
        print(f"Girdi: {X[i].tolist()} -> Tahmin: {outputs[i].item():.4f}")