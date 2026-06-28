import os
import glob
import re
import ast
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime

def load_patient_metadata(csv_path: str):
    """
    讀取 CSV 或 Excel，建立 Patient ID 對應的厚度與像素間距字典。
    """
    # 讀取檔案 (如果是 .xlsx 請改用 pd.read_excel)
    df = pd.read_csv(csv_path)
    
    metadata_dict = {}
    for index, row in df.iterrows():
        # ?? 請將下方的 'PatientID', 'SliceThickness', 'PixelSpacing' 換成你 CSV 內真實的欄位名稱！
        # 假設你的資料有欄位叫做 PatientID
        patient_id = str(row['PatientID']).strip() 
        
        # 讀取厚度
        thickness = float(row['SliceThickness'])
        
        # 讀取 Pixel Spacing
        # 假設 CSV 裡的格式是字串 "[0.976562, 0.976562]" 或用逗號分隔 "0.976,0.976"
        spacing_raw = str(row['PixelSpacing'])
        try:
            # 嘗試將字串轉為 list (例如 "[0.9, 0.9]")
            if '[' in spacing_raw:
                spacing = ast.literal_eval(spacing_raw)
            else:
                # 或是用逗號或斜線分割
                spacing = [float(x) for x in re.split(r'[,\\/]', spacing_raw)]
        except:
            # 如果解析失敗，給予一個預設值
            spacing = [1.0, 1.0]

        metadata_dict[patient_id] = {
            "thickness": thickness,
            "spacing": spacing
        }
        
    return metadata_dict

def batch_npy_to_dicom_dynamic(input_folder: str, output_root: str, csv_path: str):
    """
    自動掃描 .npy 檔案，並根據 CSV 內的資料動態設定 DICOM 厚度與間距。
    """
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    if not npy_files:
        print(f"在 '{input_folder}' 中找不到任何 .npy 檔案！")
        return

    # 1. 載入病患 Metadata 對照表
    print(f"正在載入病患空間資訊對照表: {csv_path}")
    patient_metadata = load_patient_metadata(csv_path)
    print(f"成功載入 {len(patient_metadata)} 筆病患資料。\n")

    print(f"找到 {len(npy_files)} 個 .npy 檔案，開始處理...")

    series_uid_map = {}
    study_uid_map = {}

    for npy_path in sorted(npy_files):
        filename = os.path.basename(npy_path)
        
        # 解析檔名 2BA007_slice_0154_sCT.npy
        match = re.match(r"([A-Za-z0-9]+)_slice_(\d+)_(realCT|sCT)\.npy", filename)
        if not match:
            continue

        patient_id = match.group(1)
        slice_idx = int(match.group(2))
        img_type = match.group(3)

        # 2. 動態抓取該病人的厚度與間距 (若表格中找不到，則預設給 1.0)
        if patient_id in patient_metadata:
            slice_thickness = patient_metadata[patient_id]["thickness"]
            pixel_spacing = patient_metadata[patient_id]["spacing"]
        else:
            print(f"?? 警告: 表格中找不到病患 {patient_id}，將使用預設值 Thickness=1.0, Spacing=[1.0, 1.0]")
            slice_thickness = 1.0
            pixel_spacing = [1.0, 1.0]

        folder_name = "cbct" if img_type == "sCT" else "ct"
        output_dir = os.path.join(output_root, patient_id, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # --- 處理影像矩陣 ---
        img_array = np.load(npy_path)
        img_array = np.squeeze(img_array)
        
        rescale_slope = 1
        rescale_intercept = int(img_array.min())
        if rescale_intercept >= 0:
            rescale_intercept = 0
            volume_stored = img_array.astype(np.uint16)
        else:
            volume_stored = (img_array - rescale_intercept).astype(np.uint16)

        # --- UID 產生邏輯 ---
        study_key = f"{patient_id}"
        series_key = f"{patient_id}_{img_type}"
        
        if study_key not in study_uid_map:
            study_uid_map[study_key] = generate_uid()
        if series_key not in series_uid_map:
            series_uid_map[series_key] = generate_uid()

        # --- 建立 DICOM ---
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_implicit_VR = False
        ds.is_little_endian = True

        ds.PatientName = f"Anonymous_{patient_id}"
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid_map[study_key]
        
        dt = datetime.datetime.now()
        ds.StudyDate = dt.strftime("%Y%m%d")
        ds.StudyTime = dt.strftime("%H%M%S")
        ds.SeriesInstanceUID = series_uid_map[series_key]
        ds.SeriesNumber = 1 if img_type == "sCT" else 2
        ds.Modality = "CT"
        ds.SeriesDescription = "Generated sCT (Model)" if img_type == "sCT" else "Ground Truth RealCT"

        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = slice_idx

        # 【核心修改：套用動態獲取的空間參數】
        ds.Rows = volume_stored.shape[0]
        ds.Columns = volume_stored.shape[1]
        ds.PixelSpacing = pixel_spacing
        ds.SliceThickness = slice_thickness
        
        # Z 軸位置 = 切片索引 * 該病人的真實切片厚度
        ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx) * slice_thickness]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceLocation = float(slice_idx) * slice_thickness

        # --- 寫入像素 ---
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0 
        ds.RescaleSlope = rescale_slope
        ds.RescaleIntercept = rescale_intercept
        ds.PixelData = volume_stored.tobytes()

        # 儲存
        out_path = os.path.join(output_dir, f"slice_{slice_idx:04d}.dcm")
        pydicom.dcmwrite(out_path, ds)

    print(f"\n轉換完成！所有病患的 3D 空間比例已自動校正，並儲存至：{output_root}/")

if __name__ == "__main__":
    batch_npy_to_dicom_dynamic(
        input_folder = "model_outputs",              # 放 .npy 的資料夾
        output_root  = "Brain_Model_DICOM",          # 輸出的 DICOM 資料夾
        csv_path     = "2_brain_train.xlsx - CT.csv" # 你的對照表格路徑
    )