import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import math  # 用于计算困惑度
from data import get_dataloaders
from model import Transformer

# 超参数设置（新增--use_pos_encoding参数）
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='full_with_pos', choices=['full_with_pos', 'full_without_pos'],
                    help='模型类型：full_with_pos（基础实验：带位置编码）/full_without_pos（消融实验：无位置编码）')
parser.add_argument('--data_path', type=str, default='./input.txt', help='数据集路径')
parser.add_argument('--seq_len', type=int, default=64, help='序列长度')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--d_model', type=int, default=64, help='嵌入维度')
parser.add_argument('--num_heads', type=int, default=2, help='注意力头数')
parser.add_argument('--num_layers', type=int, default=2, help='编码器/解码器层数')
parser.add_argument('--d_ff', type=int, default=256, help='前馈网络隐藏层维度')
parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪阈值')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
parser.add_argument('--use_pos_encoding', type=bool, default=True, help='是否启用位置编码（基础实验True，消融实验False）')
args = parser.parse_args()

# 创建结果保存目录
os.makedirs('./results', exist_ok=True)


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一轮（无修改）"""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Training'):
        optimizer.zero_grad()
        src, tgt = batch[0].to(device), batch[1].to(device)
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    """评估一轮（新增准确率和平均损失计算）"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            src, tgt = batch[0].to(device), batch[1].to(device)
            output = model(src, tgt[:, :-1])  # output: (batch_size, tgt_len-1, vocab_size)
            tgt_flat = tgt[:, 1:].reshape(-1)  # 目标序列：去掉第一个字符，展平为一维
            output_flat = output.reshape(-1, output.size(-1))  # 输出展平

            # 计算损失
            loss = criterion(output_flat, tgt_flat)
            total_loss += loss.item()

            # 计算准确率（忽略padding位置0）
            preds = torch.argmax(output_flat, dim=-1)
            mask = (tgt_flat != 0)  # 过滤padding
            correct = (preds == tgt_flat) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)  # 困惑度=exp(平均损失)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, perplexity, accuracy


def plot_loss(train_losses, val_losses, model_name):
    """绘制loss下降图（无修改）"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'Train Loss ({model_name})')
    plt.plot(val_losses, label=f'Val Loss ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curve ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/loss_curve_{model_name}.png')
    plt.close()


def save_metrics(metrics, model_name):
    """保存评估指标到文件"""
    with open(f'./results/metrics_{model_name}.txt', 'w', encoding='utf-8') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Final Train Loss: {metrics["train_loss"]:.4f}\n')
        f.write(f'Final Val Loss: {metrics["val_loss"]:.4f}\n')
        f.write(f'Val Perplexity: {metrics["val_perplexity"]:.4f}\n')
        f.write(f'Val Accuracy: {metrics["val_accuracy"]:.4f}\n')


def main():
    # 加载数据
    train_loader, test_loader, vocab, vocab_size = get_dataloaders(
        args.data_path, seq_len=args.seq_len, batch_size=args.batch_size
    )
    print(f'Vocab size: {vocab_size}')
    print(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    # 初始化模型（根据参数控制是否启用位置编码）
    device = torch.device(args.device)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_pos_encoding=args.use_pos_encoding  # 关键参数
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_perplexity, val_accuracy = eval_epoch(model, test_loader, criterion, device)

        # 记录loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./results/best_model_{args.model_name}.pth')

        # 打印当前指标
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')

    # 绘制loss图
    plot_loss(train_losses, val_losses, args.model_name)

    # 保存最终指标
    final_metrics = {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_perplexity': math.exp(val_losses[-1]),
        'val_accuracy': val_accuracy
    }
    save_metrics(final_metrics, args.model_name)

    print(f'\nTraining finished!')
    print(f'Loss curve saved to ./results/loss_curve_{args.model_name}.png')
    print(f'Metrics saved to ./results/metrics_{args.model_name}.txt')


if __name__ == '__main__':
    main()