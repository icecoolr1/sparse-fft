import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)


    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # FFT-based forecasting
    # 对时间维度（period_len）进行 FFT
        X_fft = torch.fft.fft(x, dim=1)

        # 处理频域数据，例如截断高频分量
        k = int(self.period_len * 0.5)  # 保留前一半的低频分量
        X_fft_truncated = X_fft.clone()
        X_fft_truncated[:, k:, :] = 0  # 截断高频

        # 逆 FFT 得到时域预测
        y_ifft = torch.fft.ifft(X_fft_truncated, dim=1).real  # 取实部
        
        y = y_ifft.permute(0, 2, 1).reshape(batch_size, -1, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y
