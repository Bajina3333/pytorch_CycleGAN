import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

# 1. 設定您的 log 檔案路徑
log_path = './checkpoints/unet2plus_run/loss_log.txt' 
data = []

print(f"正在讀取並解析 Log 檔案: {log_path} ...")
try:
    with open(log_path, 'r') as f:
        for line in f:
            if '(epoch:' in line:
                try:
                    iters = int(re.search(r'iters:\s*(\d+)', line).group(1))
                    epoch = int(re.search(r'epoch:\s*(\d+)', line).group(1))
                    
                    parts = line.split(') , ')
                    if len(parts) > 1:
                        # 精準抓取右半部數值
                        loss_str = parts[1].strip() 
                        
                        loss_pairs = loss_str.split(', ')
                        losses = {'epoch': epoch, 'iters': iters}
                        
                        for pair in loss_pairs:
                            if ': ' in pair:
                                key, val = pair.split(': ')
                                losses[key] = float(val)
                                
                        data.append(losses)
                except Exception as e:
                    continue
except FileNotFoundError:
    print(f"找不到檔案！請確認 {log_path} 這個路徑是否正確。")
    sys.exit(1)

# 2. 轉換成 Pandas DataFrame
df = pd.DataFrame(data)

if df.empty:
    print("錯誤：成功讀取檔案，但沒有解析出任何 Loss 數據！請檢查檔案內容格式。")
    sys.exit(1)

df['Total_Iters'] = range(1, len(df) + 1)

# 找出哪些是 Paired/L1 Loss
paired_cols = [col for col in df.columns if 'L1' in col]

# ==========================================
# 🌟 第一張圖：標準 CycleGAN 核心誤差
# ==========================================
plt.figure(figsize=(15, 6)) # 高度稍微調扁一點，適合並排放在報告中
if 'G_A' in df.columns:
    plt.plot(df['Total_Iters'], df['G_A'], label='G_A Loss (Generator)', alpha=0.7, linewidth=1.5)
if 'cycle_A' in df.columns:
    plt.plot(df['Total_Iters'], df['cycle_A'], label='Cycle_A Loss (Reconstruction)', alpha=0.8, linewidth=1.5)
if 'D_A' in df.columns:    
    plt.plot(df['Total_Iters'], df['D_A'], label='D_A Loss (Discriminator)', alpha=0.5, linewidth=1.5)

plt.title('CycleGAN Standard Training Loss (Generator & Cycle)')
plt.xlabel('Logged Steps (across all epochs)')
plt.ylabel('Loss Value')
plt.ylim(0, 2.0)  # 保留 0~2.5 的限制，讓 GAN 的細節清晰可見
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path_standard = './results/unet2plus_run/loss_convergence.png'
plt.savefig(save_path_standard, dpi=300)
plt.close() # 關閉畫布避免與下一張圖重疊

print(f"標準誤差圖繪圖完成: {save_path_standard}")

# ==========================================
# 🌟 第二張圖：專屬 Paired L1 Loss 誤差圖 (如果有的話)
# ==========================================
if paired_cols:
    plt.figure(figsize=(15, 6))
    colors = ['red', 'blue']
    i=0
    for col in paired_cols:
        c = colors[i % len(colors)]
        # 畫出淡淡的原始震盪背景
        plt.plot(df['Total_Iters'], df[col], label=f'{col}', color=c, alpha=0.7, linewidth=1)
        
        i=i+1
    plt.title('CycleGAN Paired L1 Loss Convergence')
    plt.xlabel('Logged Steps (across all epochs)')
    plt.ylabel('Paired L1 Loss Value')
    plt.ylim(0, 4.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path_paired = './results/unet2plus_run/loss_convergence.png'
    plt.savefig(save_path_paired, dpi=300)
    plt.close()
    
    print(f"Paired 專屬圖繪圖完成: {save_path_paired}")
else:
    print(f"繪圖完成！未偵測到 Paired L1 Loss，僅產出標準誤差圖: {save_path_standard}")