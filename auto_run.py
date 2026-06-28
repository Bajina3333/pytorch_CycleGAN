import os
import time
import subprocess

# 設定需求(跟自個人需要的記憶體大小，這裡以 19 GB為範例)
REQUIRED_MEMORY = 19500  # 啟動所需的最小空閒記憶體 (MB)
TARGET_GPU = 0           # 想排隊等待的顯卡編號 (0 是 RTX 4090，1 是 RTX 3090)
CHECK_INTERVAL = 30      # 每隔幾秒檢查一次 (60秒)

def get_free_memory(gpu_id):
    """呼叫 nvidia-smi 獲取指定顯卡的剩餘記憶體"""
    try:
        cmd = f"nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader -i {gpu_id}"
        result = subprocess.check_output(cmd, shell=True)
        return int(result.decode('utf-8').strip())
    except Exception as e:
        print(f"獲取 GPU 資訊失敗: {e}")
        return 0

print(f"開始自動排隊監控...")
print(f"正在等待 GPU {TARGET_GPU} 釋放超過 {REQUIRED_MEMORY} MB 的空間...")

while True:
    free_mem = get_free_memory(TARGET_GPU)
    
    if free_mem >= REQUIRED_MEMORY:
        print(f"\n 偵測到 GPU {TARGET_GPU} 已釋放空間！目前空閒: {free_mem} MB")
        print("立即啟動接續訓練程式...")
        
        # 這裡放入實際要跑的訓練指令，這裡以"python train.py --dataroot ./datasets/cbct2ct --name medical_run"為例 (已加上指定顯卡的參數)
        train_command = (
            f"CUDA_VISIBLE_DEVICES={TARGET_GPU} python train.py --dataroot ./datasets/cbct2ct --name unet2plus_run --netG liang_unet2plus"
        )
        
        # 執行訓練
        os.system(train_command)
        break  # 訓練啟動（或結束）後，跳出監控迴圈
    else:
        # 使用 \r 讓輸出保持在同一行覆蓋，畫面才不會被洗版
        print(f"目前 GPU {TARGET_GPU} 僅剩 {free_mem} MB 空閒，繼續等待中...", end="\r")
        time.sleep(CHECK_INTERVAL)