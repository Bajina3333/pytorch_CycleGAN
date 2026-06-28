import os
import glob
import shutil

# ================= 路徑設定 (使用絕對路徑) =================
# 同事的共享資料夾 (來源：讀取 DICOM)
SOURCE_ROOT = "/data/3THDD/dataset/CBCT2CT/brain_DICOM/"

# CycleGAN 模型資料夾下的 datasets 目錄 (目標：寫入整理好的檔案)
TARGET_ROOT = "/data/1THDD/data/TJLiao/pytorch_CycleGAN/datasets/cbct2ct"
# =========================================================

def organize_dataset():
    print(f"開始從 {SOURCE_ROOT} 讀取資料...")
    print(f"預計輸出至 {TARGET_ROOT} ...")
    
    # 建立 CycleGAN 所需的資料夾 (包含 train, val, test)
    for phase in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_ROOT, f"{phase}A"), exist_ok=True)
        os.makedirs(os.path.join(TARGET_ROOT, f"{phase}B"), exist_ok=True)

    patients = os.listdir(SOURCE_ROOT)
    
    for patient_id in patients:
        patient_path = os.path.join(SOURCE_ROOT, patient_id)
        if not os.path.isdir(patient_path):
            continue

        # ================= 切分名單設定 =================
        # 可以把需要做 Validation 與 Test 的病患 ID 填入對應的列表中
        test_patients = ['2BA007', '2BA014', '2BA028', '2BA035', '2BA049', '2BA056', '2BA063', '2BA077', '2BB007', '2BB011', '2BB017', '2BB028', '2BB049', '2BB075', '2BB109', '2BB151', '2BB175', '2BB179', '2BC007', '2BC014', '2BC021', '2BC042',  '2BC070']
        val_patients  = ['2BA037', '2BA044', '2BA088', '2BB034', '2BB083', '2BB096', '2BB198', '2BC016', '2BC027', '2BC037', '2BC056', '2BC081'] 
        
        if patient_id in test_patients:
            phase = 'test'
        elif patient_id in val_patients:
            phase = 'val'
        else:
            phase = 'train'
        # ================================================

        # 抓取該病患的所有 CBCT 與 CT 切片
        cbct_slices = sorted(glob.glob(os.path.join(patient_path, "cbct", "*.dcm")))
        ct_slices = sorted(glob.glob(os.path.join(patient_path, "ct", "*.dcm")))

        # 確保成對的切片數量一致
        if len(cbct_slices) != len(ct_slices):
            print(f"警告：病患 {patient_id} 的 CBCT ({len(cbct_slices)}) 與 CT ({len(ct_slices)}) 切片數量不一致！已跳過。")
            continue

        # 複製並重新命名檔案 (例如: 2BA015_slice_001.dcm)
        for cbct_path, ct_path in zip(cbct_slices, ct_slices):
            slice_name = os.path.basename(cbct_path)
            new_filename = f"{patient_id}_{slice_name}"
            
            # 將 CBCT 放進 phaseA，CT 放進 phaseB
            shutil.copy(cbct_path, os.path.join(TARGET_ROOT, f"{phase}A", new_filename))
            shutil.copy(ct_path, os.path.join(TARGET_ROOT, f"{phase}B", new_filename))
            
        print(f"  -> 病患 {patient_id} 處理完成，分配至: {phase} 集")
            
    print("\n所有資料集分類與整理完成！")

if __name__ == "__main__":
    organize_dataset()
    
    