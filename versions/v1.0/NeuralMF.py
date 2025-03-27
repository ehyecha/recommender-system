# Neural MF Model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class NeuralMF(pl.LightningModule):
    def __init__(self, num_users, num_items, num_numeric_features, latent_dim, learning_rate=0.001):
        super(NeuralMF, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.numeric_fc = nn.Linear(num_numeric_features, latent_dim)
        # Neural network layers
        self.fc1 = nn.Linear(latent_dim * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, user, item, numeric_features):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        numeric_emb = self.numeric_fc(numeric_features)
        # Concatenate user and item embeddings
        x = torch.cat([user_embed, item_embed, numeric_emb], dim=1)
        #Feed through fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)  # ✅ Dropout 적용
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return (torch.sigmoid(x) * 5).squeeze()
    
    def training_step(self, batch, batch_idx):
        user_ids, item_ids, numeric_features, ratings = batch
        predicted_ratings = self.forward(user_ids, item_ids, numeric_features)
        loss = self.criterion(predicted_ratings, ratings)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True,  on_step = True)  # 🔥 손실 로깅 추가
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, numeric_features, ratings = batch
        predicted_ratings = self.forward(user_ids, item_ids, numeric_features)
        loss = self.criterion(predicted_ratings, ratings)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True, on_step = True)  # 🔥 손실 로깅 추가
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Lightning에서 `trainer.predict()`를 호출할 때 사용"""
        user, item, numeric_features, ratings = batch  # 배치에서 올바른 입력 추출
        prediction = self.forward(user, item, numeric_features)
        return prediction, ratings
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate,  weight_decay=1e-3)