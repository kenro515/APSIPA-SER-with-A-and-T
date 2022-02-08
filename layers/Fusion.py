import torch
import torch.nn as nn

torch.manual_seed(1234)


class EarlyConcat_LateLinear(nn.Module):
    def __init__(self, in_text_dim=768, in_audio_dim=256, out_dim=4, dropout=0.5):
        super(EarlyConcat_LateLinear, self).__init__()

        self.linear_audio = nn.Linear(in_audio_dim, in_audio_dim)

        self.linear_1 = nn.Linear(in_text_dim, in_audio_dim)
        self.linear_2 = nn.Linear(
            (in_audio_dim + in_audio_dim), in_audio_dim)
        self.linear_3 = nn.Linear(in_audio_dim, out_dim)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_text, x_audio):
        x_audio = self.linear_audio(x_audio)
        x_audio_activation = self.activation(x_audio)
        x_audio_dropout = self.dropout(x_audio_activation)

        x1 = self.linear_1(x_text)
        x1_activation = self.activation(x1)
        x1_dropout = self.dropout(x1_activation)

        x_cat = torch.cat([x_audio_dropout, x1_dropout], dim=1)

        x2 = self.linear_2(x_cat)
        x2_activation = self.activation(x2)
        x2_dropout = self.dropout(x2_activation)

        x3 = self.linear_3(x2_dropout)

        return x3


class EarlyMul_LateLinear(nn.Module):
    def __init__(self, in_text_dim=768, in_audio_dim=256, out_dim=4, dropout=0.5):
        super(EarlyMul_LateLinear, self).__init__()

        self.linear_audio = nn.Linear(in_audio_dim, in_audio_dim)

        self.linear_1 = nn.Linear(in_text_dim, in_audio_dim)
        self.linear_2 = nn.Linear(in_audio_dim, out_dim)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_text, x_audio):
        x_audio = self.linear_audio(x_audio)
        x_audio_activation = self.activation(x_audio)
        x_audio_dropout = self.dropout(x_audio_activation)

        x1 = self.linear_1(x_text)
        x1_activation = self.activation(x1)
        x1_dropout = self.dropout(x1_activation)

        x_cat = torch.mul(x_audio_dropout, x1_dropout)

        x2 = self.linear_2(x_cat)

        return x2


class EarlyConcat_LateFC(nn.Module):
    def __init__(self, in_text_dim=768, in_audio_dim=256, out_dim=4, dropout=0.5):
        super(EarlyConcat_LateFC, self).__init__()

        self.linear_audio = nn.Linear(in_audio_dim, in_audio_dim)

        self.linear_1 = nn.Linear(in_text_dim, in_audio_dim)
        self.linear_2 = nn.Linear(
            (in_audio_dim + in_audio_dim), in_audio_dim)
        self.linear_3 = nn.Linear(in_audio_dim, out_dim)

        self.linear_out = nn.Linear(out_dim * 3, out_dim)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_text, x_audio, pred_text, pred_audio):
        x_audio = self.linear_audio(x_audio)
        x_audio_activation = self.activation(x_audio)
        x_audio_dropout = self.dropout(x_audio_activation)

        x1 = self.linear_1(x_text)
        x1_activation = self.activation(x1)
        x1_dropout = self.dropout(x1_activation)

        x_early = torch.cat([x_audio_dropout, x1_dropout], dim=1)

        x2 = self.linear_2(x_early)
        x2_activation = self.activation(x2)
        x2_dropout = self.dropout(x2_activation)

        x3 = self.linear_3(x2_dropout)

        x_late = torch.cat([x3, pred_text, pred_audio], dim=1)
        output = self.linear_out(x_late)

        return output


class EarlyMul_LateFC(nn.Module):
    def __init__(self, in_text_dim=768, in_audio_dim=256, out_dim=4, dropout=0.5):
        super(EarlyMul_LateFC, self).__init__()

        self.linear_audio = nn.Linear(in_audio_dim, in_audio_dim)

        self.linear_1 = nn.Linear(in_text_dim, in_audio_dim)
        self.linear_2 = nn.Linear(in_audio_dim, out_dim)

        self.linear_out = nn.Linear(out_dim * 3, out_dim)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_text, x_audio, pred_text, pred_audio):
        x_audio = self.linear_audio(x_audio)
        x_audio_activation = self.activation(x_audio)
        x_audio_dropout = self.dropout(x_audio_activation)

        x1 = self.linear_1(x_text)
        x1_activation = self.activation(x1)
        x1_dropout = self.dropout(x1_activation)

        x_early = torch.mul(x_audio_dropout, x1_dropout)

        x2 = self.linear_2(x_early)

        x_late = torch.cat([x2, pred_text, pred_audio], dim=1)
        output = self.linear_out(x_late)

        return output
