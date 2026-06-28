import os
import torch
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as calculate_ssim

# 引入 CycleGAN 官方的選項與模型讀取模組
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# 2D 版本
def compute_similarity_metrics(sCT_hu, real_CT_hu):
    """依照 Liang 2019 論文公式計算指標"""
    # 1. MAE & RMSE
    mae = np.mean(np.abs(sCT_hu - real_CT_hu))
    mse = np.mean((sCT_hu - real_CT_hu) ** 2)
    rmse = np.sqrt(mse)

    # 2. PSNR & SSIM (論文指定動態範圍為 4071)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(4071.0 / rmse)
        
    ssim_value = calculate_ssim(sCT_hu, real_CT_hu, data_range=4071.0)
    
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

    results = []

    # 設定與您前處理時完全一致的 HU 值正規化範圍
    HU_MIN = -1000.0
    HU_MAX = 3000.0

    print("開始進行推論、匯出二進位檔與指標計算...")

    # 3. 進入測試迴圈
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

        # 呼叫函數計算指標
        mae, rmse, ssim_val, psnr_val = compute_similarity_metrics(fake_ct_hu, real_ct_hu)

        # 將計算結果儲存至列表
        results.append({
            'PatientID': patient_id,
            'Filename': filename,
            'MAE': mae,
            'RMSE': rmse,
            'SSIM': ssim_val,
            'PSNR': psnr_val
        })

        if i % 100 == 0:
            print(f"已處理 {i} 張切片... (最新檔名: {filename})")

    # 4. 彙整結果並產出 Excel 報表
    df = pd.DataFrame(results)
    
    # 建立一個新的列表，用來排版「切片 -> 單一病患平均 -> 總平均」
    final_report_data = []

    # 利用 Pandas 的 groupby 功能，自動依照病患 ID (PatientID) 分組
    for pid, group in df.groupby('PatientID'):
        # 1. 先把該病患的「所有個別切片」資料加進去
        final_report_data.extend(group.to_dict('records'))
        
        # 2. 計算該病患的「獨立平均值」，並加入一個醒目的分隔列
        final_report_data.append({
            'PatientID': pid,
            'Filename': f"{pid} 切片平均",
            'MAE': group['MAE'].mean(),
            'RMSE': group['RMSE'].mean(),
            'SSIM': group['SSIM'].mean(),
            'PSNR': group['PSNR'].mean()
        })

    # 3. 計算所有切片的「總平均」
    # (注意：對原始 df 算總平均，才不會因病患切片數不同導致失真)
    final_report_data.append({
        'PatientID': 'ALL',
        'Filename': f"所有病患總平均",
        'MAE': df['MAE'].mean(),
        'RMSE': df['RMSE'].mean(),
        'SSIM': df['SSIM'].mean(),
        'PSNR': df['PSNR'].mean()
    })

    # 轉換成最終的 DataFrame
    final_df = pd.DataFrame(final_report_data)
    
    # 把剛剛作為分組輔助用的 'PatientID' 欄位刪除，讓 Excel 畫面保持乾淨
    final_df = final_df.drop(columns=['PatientID'])

    # 匯出至 Excel
    excel_path = os.path.join(excel_export_dir, 'Evaluation_Metrics.xlsx')
    final_df.to_excel(excel_path, index=False)

    print("=========================================================")
    print(f"評估與匯出、NumPy 二進位檔案 (.npz)、Excel 四項指標報表皆已完成")
    print("=========================================================")