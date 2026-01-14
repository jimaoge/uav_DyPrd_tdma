from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleLSTM(nn.Module):
    """简单LSTM，用于单步预测"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)  # pred(batch_size, seq_len, output_size)
        pred = pred[:, -1, :]  # 只取最后一步
        return pred


class Encoder(nn.Module):
    """编码器 - 处理历史序列"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    """解码器 - 生成未来序列"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        # x: [batch, 1, input_size]
        # hidden, cell: [num_layers, batch, hidden_size]
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # [batch, output_size]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """完整的seq2seq模型，使用Encoder和Decoder"""
    def __init__(self, encoder, decoder, device, pred_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pred_len = pred_len
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: 源序列 [batch, src_len, input_size]
            tgt: 目标序列 [batch, tgt_len, output_size] (训练时)
            teacher_forcing_ratio: teacher forcing比例
        Returns:
            predictions: 预测序列 [batch, tgt_len, output_size]
        """
        batch_size = src.shape[0]
        
        # 编码
        hidden, cell = self.encoder(src)
        
        # 解码器初始输入
        decoder_input = src[:, -1:, :]  # 取最后一步
        predictions = []
        
        for t in range(self.pred_len):
            # 解码一步
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions.append(prediction.unsqueeze(1))  # [batch, 1, output_size]
            
            # 决定下一步输入
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: 使用真实值
                decoder_input = tgt[:, t:t+1, :]
            else:
                # 使用预测值
                decoder_input = prediction.unsqueeze(1)
        
        return torch.cat(predictions, dim=1)  # [batch, tgt_len, output_size]


class TopologyPredictor(nn.Module):
    """针对拓扑预测的完整模型"""
    def __init__(self, input_size, hidden_size, num_layers, pred_len, output_size=None, dropout=0.1, device='cuda'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.output_size = output_size or input_size
        self.device = device
        
        # 创建编码器和解码器
        encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        decoder = Decoder(input_size, hidden_size, num_layers, self.output_size, dropout)
        
        # Seq2Seq模型
        self.model = Seq2Seq(encoder, decoder, device, pred_len)
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        return self.model(src, tgt, teacher_forcing_ratio)
    
    def predict(self, src, pred_len=None):
        """推理模式"""
        self.eval()
        with torch.no_grad():
            if pred_len is None:
                pred_len = self.pred_len
            return self.forward(src, tgt=None, teacher_forcing_ratio=0)


# 使用示例
if __name__ == "__main__":
    # 参数
    INPUT_SIZE = 3  # xyz位置
    HIDDEN_SIZE = 128   # LSTM隐藏层维度，决定模型记忆和表达能力
    NUM_LAYERS = 2      # LSTM堆叠层数，增加网络深度，层数越多，模型表示能力越强，
    PRED_LEN = 12       # 预测序列长度，即要预测的未来时间步数
    BATCH_SIZE = 16     # 批量大小，每次训练时输入模型的样本数
    SEQ_LEN = 24        # 输入序列长度，历史观测时间步数
    
    # 创建模型
    model = TopologyPredictor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN
    ).to(device)
    
    # 测试数据
    src_seq = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE).to(device)
    tgt_seq = torch.randn(BATCH_SIZE, PRED_LEN, INPUT_SIZE).to(device)
    
    # 前向传播
    # 训练模式
    output_train = model(src_seq, tgt_seq, teacher_forcing_ratio=0.7)
    print(f"训练输出形状: {output_train.shape}")  # [16, 10, 6]
    
    # 推理模式
    output_infer = model.predict(src_seq)
    print(f"推理输出形状: {output_infer.shape}")  # [16, 10, 6]
    
    # 计算损失
    criterion = nn.MSELoss()
    loss = criterion(output_train, tgt_seq)
    print(f"训练损失: {loss.item():.4f}")