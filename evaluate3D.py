import os
import torch
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as calculate_ssim

# 引入 CycleGAN 官方的選項與模型讀取模組
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# 改為 3D 體積版本
def compute_similarity_metrics_3d(sCT_vol, real_CT_vol):
    """依照 Liang 2019 論文公式以 3D 體積計算指標"""
    # 1. MAE & RMSE (直接對整個 3D 矩陣算平均)
    mae = np.mean(np.abs(sCT_vol - real_CT_vol))
    mse = np.mean((sCT_vol - real_CT_vol) ** 2)
    rmse = np.sqrt(mse)

    # 2. PSNR & SSIM (論文指定動態範圍為 4071)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(4071.0 / rmse)
        
    # channel_axis=None 確保輸入 (D, H, W) 被當作單通道的 3D 體積影像進行計算
    ssim_value = calculate_ssim(sCT_vol, real_CT_vol, data_range=4071.0, channel_axis=None)
    
    return mae, rmse, ssim_value, psnr


if __name__ == '__main__':
    # 1. 解析測試參數與初始化設定
    opt = TestOptions().parse()

    # 手動指定運算裝置 (解決舊版程式碼 AttributeError 的問題)
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 鎖定測試時的參數，確保結果穩定與正確性
    opt.num_threads = 0        # 測試模式鎖定單執行緒
    opt.batch_size = 1         # 每次處理 1 張切片
    opt.serial_batches = True  # 依序讀取不打亂
    opt.no_flip = True         # 嚴禁翻轉影像

    # 建立資料集與模型
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    # 建立匯出目錄
    # 存放 .npz 二進位檔案的資料夾
    binary_export_dir = os.path.join(opt.results_dir, opt.name, 'npy_exports')
    os.makedirs(binary_export_dir, exist_ok=True)
    
    # 存放 Excel 報表的資料夾
    excel_export_dir = os.path.join(opt.results_dir, opt.name)
    os.makedirs(excel_export_dir, exist_ok=True)

    # 用於暫存每位病患所有切片矩陣的字典，以便後續堆疊成 3D 體積
    patient_slices_fake = {}
    patient_slices_real = {}

    # 設定與您前處理時完全一致的 HU 值正規化範圍
    HU_MIN = -1000.0
    HU_MAX = 3000.0

    print("開始進行推論、匯出二進位檔...")

    # 3. 進入測試迴圈 (此處不變動，維持 2D 切片推論、反正規化與 .npy 存檔)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()  # 執行前向推論

        # 取得檔名並提取病患 ID
        img_path = model.get_image_paths()[ 0 ]
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[ 0 ] # 例如: 2BA015_slice001
        patient_id = base_name.split('_')[ 0 ]      # 切割並只取第一部分，例如: 2BA015

        # 從顯卡攔截未壓縮的浮點數張量，壓平為 2D (512, 512)，並轉回 CPU 的 NumPy 格式
        fake_B_tensor = model.fake_B.squeeze().detach().cpu().numpy()
        real_B_tensor = model.real_B.squeeze().detach().cpu().numpy()

        # 反正規化：將模型輸出的 [-1, 1] 還原成真實的 HU 值 [-1000, 3000]
        fake_ct_hu = (fake_B_tensor + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN
        real_ct_hu = (real_B_tensor + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN

        # .npy 獨立檔案 (自動依病患分類資料夾)
        # 1. 建立該病患專屬的子資料夾 (exist_ok=True 確保資料夾存在時不會報錯)
        patient_dir = os.path.join(binary_export_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # 2. 設定獨立的 .npy 檔案路徑
        fake_export_path = os.path.join(patient_dir, f"{base_name}_sCT.npy")
        real_export_path = os.path.join(patient_dir, f"{base_name}_realCT.npy")
        
        # 3. 使用 np.save 獨立存檔
        np.save(fake_export_path, fake_ct_hu)
        np.save(real_export_path, real_ct_hu)

        # 將切片加入字典，以便後續組合 3D 體積 (依據讀取順序自動按 Z 軸排序)
        if patient_id not in patient_slices_fake:
            patient_slices_fake[patient_id] = []
            patient_slices_real[patient_id] = []
        
        patient_slices_fake[patient_id].append(fake_ct_hu)
        patient_slices_real[patient_id].append(real_ct_hu)

        if i % 100 == 0:
            print(f"已處理 {i} 張切片... (最新檔名: {filename})")

    print("\n所有切片推論與存檔完成。開始將切片組合為 3D 體積並計算 3D 指標...")

    # 4. 彙整結果並產出 Excel 報表 (改為以 3D 體積為單位計算)
    results_3d = []

    # 依病患 ID 迴圈，將各自的所有切片堆疊成 3D 體積進行評估
    for pid in patient_slices_fake.keys():
        # 將 2D 切片列表堆疊成 3D 矩陣，形狀為 (Depth, Height, Width)
        vol_fake = np.stack(patient_slices_fake[pid], axis=0)
        vol_real = np.stack(patient_slices_real[pid], axis=0)

        # 呼叫 3D 指標計算函數
        mae, rmse, ssim_val, psnr_val = compute_similarity_metrics_3d(vol_fake, vol_real)

        # 將 3D 計算結果儲存至列表
        results_3d.append({
            'Filename': f"{pid} 3D體積平均",
            'MAE': mae,
            'RMSE': rmse,
            'SSIM': ssim_val,
            'PSNR': psnr_val
        })
        print(f"病患 {pid} 3D 體積計算完成 (切片數: {vol_fake.shape[0]})")

    # 轉換成 DataFrame
    df_3d = pd.DataFrame(results_3d)

    # 建立報表列表並加入所有病患的 3D 總平均
    final_report_data = list(results_3d)
    final_report_data.append({
        'Filename': f"所有病患 3D 總平均",
        'MAE': df_3d['MAE'].mean(),
        'RMSE': df_3d['RMSE'].mean(),
        'SSIM': df_3d['SSIM'].mean(),
        'PSNR': df_3d['PSNR'].mean()
    })

    # 轉換成最終的 DataFrame 並導出 Excel
    final_df = pd.DataFrame(final_report_data)
    excel_path = os.path.join(excel_export_dir, 'Evaluation_3D.xlsx')
    final_df.to_excel(excel_path, index=False)

    print("=========================================================")
    print(f"評估與匯出、NumPy 二進位檔案 (.npy)、Excel 3D體積四項指標報表皆已完成")
    print("=========================================================")