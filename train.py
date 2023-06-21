import torch, torch.nn as nn, torchvision as tv
import lightning as L

train = tv.datasets.MNIST('.', train=True, download=True, transform=tv.transforms.ToTensor())
val = tv.datasets.MNIST('.', train=False, download=True, transform=tv.transforms.ToTensor())

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(True), nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 32), nn.ReLU(True))
        self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(True), nn.Linear(64, 128), nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Sigmoid(), nn.Unflatten(1, (28,28)))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.flatten(self.decoder(self.encoder(x))), self.flatten(x))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.flatten(self.decoder(self.encoder(x))), self.flatten(x))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


autoencoder = LitAutoEncoder()
trainer = L.Trainer()
trainer.fit(autoencoder, torch.utils.data.DataLoader(train, batch_size=256, shuffle=True), torch.utils.data.DataLoader(val, batch_size=256))
