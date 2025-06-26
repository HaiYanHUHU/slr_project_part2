import torch
import torch.nn as nn

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=960, hidden_dim=256, num_classes=300, num_layers=2):
        # BiLSTM Attention model is used for sequence feature modeling
        super(BiLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [B, T, D]
        lstm_out, _ = self.lstm(x)  #[B, T, 2H]
        attn_scores = self.attention(lstm_out)  # [B, T, 1]
        
        # Time dimension normalization
        attn_weights = torch.softmax(attn_scores, dim=1)  
        weighted = torch.sum(attn_weights * lstm_out, dim=1)  # [B, 2H]
        logits = self.classifier(weighted)  # [B, C]
        return logits, attn_weights
