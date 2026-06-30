import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os

# 建立輸出資料夾
output_dir = './results/unet2plus_run'
os.makedirs(output_dir, exist_ok=True)

# 設定您的 log 檔案路徑
log_path = './checkpoints/unet2plus_run/loss_log.txt'
data = []

print(f"正在讀取並解析 Log 檔案: {log_path} ...")
try:
    with open(log_path, 'r') as f:
        for line in f:
            if '(epoch:' in line:
                try:
                    # 抓取 epoch 數值
                    epoch = int(re.search(r'epoch:\s*(\d+)', line).group(1))
                    entry = {'epoch': epoch}
                    
                    # 自動抓取該行所有的 Loss 項目 (例如 G_A: 0.655)
                    # 排除 epoch, iters, time, data 等非 loss 數值
                    metrics = re.findall(r'([A-Za-z0-9_]+):\s*([\d.]+)', line)
                    for name, value in metrics:
                        if name not in ['epoch', 'iters', 'time', 'data']:
                            entry[name] = float(value)
                            
                    data.append(entry)
                except Exception as e:
                    continue
except FileNotFoundError:
    print(f"找不到檔案！請確認 {log_path} 這個路徑是否正確。")
    sys.exit(1)

# 轉換成 Pandas DataFrame 並「按 Epoch 取平均」
df = pd.DataFrame(data)

if df.empty:
    print("錯誤：成功讀取檔案，但沒有解析出任何 Loss 數據！請檢查檔案內容格式。")
    sys.exit(1)

# 將所有相同的 epoch 群組起來，並計算其算術平均數
df_epoch_avg = df.groupby('epoch').mean().reset_index()

# 找出哪些是 Paired/L1 Loss
paired_cols = [col for col in df_epoch_avg.columns if 'L1' in col]

#==========================================
# 繪製平滑的 Epoch 平均誤差圖
#==========================================
plt.figure(figsize=(15, 6))

# 畫出標準的 CycleGAN Loss (加粗線條 linewidth=2 讓趨勢更明顯)
if 'G_A' in df_epoch_avg.columns:
    plt.plot(df_epoch_avg['epoch'], df_epoch_avg['G_A'], label='G_A Loss (Generator)', linewidth=2)
if 'D_A' in df_epoch_avg.columns:
    plt.plot(df_epoch_avg['epoch'], df_epoch_avg['D_A'], label='D_A Loss (Discriminator)', linewidth=2)

# 如果有 Paired L1 Loss，用虛線畫出來以作區別
for col in paired_cols:
    plt.plot(df_epoch_avg['epoch'], df_epoch_avg[col], label=f'{col} (Smoothed Paired Loss)', linewidth=2, linestyle='--')

plt.title('CycleGAN Training Loss Convergence (Averaged by Epoch)')
plt.xlabel('Epoch') # X 軸從 Total Iters 改為 Epoch
plt.ylabel('Average Loss Value')

# 可以根據實際跑出來的圖表，調整 ylim。如果最高只有 1.0，可以把 2.0 改小。
plt.ylim(0, 2.0)  

plt.legend()
plt.grid(True)
plt.tight_layout()

# 儲存圖表
save_path_smooth = os.path.join(output_dir, 'loss_convergence_smooth.png')
plt.savefig(save_path_smooth, dpi=300)
plt.close()

print(f"平滑的 Epoch 平均誤差圖繪製完成: {save_path_smooth}")