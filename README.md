# pytorch_CycleGAN
本專案基於原始 PyTorch CycleGAN 框架進行深度改動，旨在實現精準的醫學影像翻譯（CBCT-to-CT）。專案的初期架構靈感與參數設定參考自醫學期刊論文 Liang 等人(2019)，並在後續階段針對配對資料集的優勢進行了 Loss Function 的改寫，目前正積極導入更深層的 UNet++ 架構以期獲得更極致的解剖結構保留效果。

> 原始 CycleGAN 架構：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git <br/>
> 出自論文：https://doi.org/10.48550/arXiv.1703.10593 和 https://doi.org/10.48550/arXiv.1611.07004 <br/>
> 初期架構參考論文：https://doi.org/10.48550/arXiv.1810.13350

# Project Stages
本專案的演進主要分為以下幾個階段：
> 註：僅做 HU 值截斷至 [-1000,3000] 與歸一化至 [-1,1]、尺寸校正為 512x512。<br/>
- Stage 1：（針對模型本身的改動）
  - Phase 1：對齊 Liang(2019) 論文架構
  - Phase 2：活用配對資料集優勢
  - Phase 3：生成器升級為 UNet++ (進行中)
- Stage 2：（針對資料集做前處理，以 Phase 1 的模型做訓練）
  - Phase 4：消除 CBCT 的誤差 (進行中)

## Phase 1
本階段的核心目標是捨棄官方預設的「通用型轉換」架構，將模型打造成符合醫療影像高精度需求的專屬架構。

- Generator 重構：
  - 改寫為醫療影像專用的 U-Net 架構。將下採樣次數精確設定為 5 次（即 8 層卷積），使得 512×512 的輸入在最深處的瓶頸層尺寸保留為 16×16×512，並在瓶頸層使用 1×1 卷積。
  - 基礎特徵通道數 (ngf) 從預設的 64 調整為 32。
  - 消除棋盤格偽影：底層卷積捨棄了容易產生偽影的反卷積 (ConvTranspose2d)，改為交替使用 Stride 1 與 Stride 2 的卷積，並搭配 雙線性插值 (Bilinear Upsampling) 進行上採樣。
  - 激勵函數替換：所有層在 Instance Normalization 之後，全面採用斜率 0.2 的 LeakyReLU，以保留微弱的醫療解剖特徵，並強制關閉 Dropout。

- Discriminator 升級：
  - 針對高解析度影像，將網路擴展為 4 層，打造出感受野為 142×142 的 PatchGAN。

- Loss Function 與訓練參數設定：
  - 嚴格鎖定 Cycle Loss 權重為 10，Identity Loss 權重為 5 (lambda_identity=0.5)，確保解剖位置不發生偏移。優化器採用 Adam，學習率固定為 0.0002 且 β1=0.5。
 
- 訓練測試：
  - 在此配置下，已經成功完成了 50 Epochs 與 200 Epochs 的訓練與效果比較。<br/>

- Loss Curve 走勢圖：<br/>
  - Epochs 50：
![Epochs 50](/results/medical_run/loss_convergence_smooth.png)
  - Epochs 200：
![Epochs 200](/results/medical_run_200/loss_convergence_smooth.png)

- 數據比較：進行 Liang(2019) 論文數據與分別經過 50 Epochs 和 200 Epochs 的訓練後數據的比較。

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |
| Epoch_50 | 24.27 ± 5.64 | 92.51 ± 18.27 | 0.95 ± 0.01 | 33.06 ± 1.92 |
| Epoch_200 | 14.69 ± 4.13 | 75.65 ± 20.41 | 0.96 ± 0.02 | 34.91 ± 2.29 |
<br/>

- 實際圖片比較：<br/>
> 完整檔案為 /results 的 Medical.html<br/>

![Medical.png](/results/html_img/Medical.png)

## Phase 2
在確保無監督翻譯能力的基礎上，進一步思考如何利用既有的「配對資料集 (Paired Dataset)」來強化模型表現。

- 引入 Paired L1 Loss：
  - 在原有的 CycleGAN 損失函數中，額外加入了直接針對真實 CT 與生成 sCT 計算像素級絕對誤差的 Paired L1 Loss 機制，強迫模型在像素級別與 Ground Truth 對齊。

- 彈性資料讀取機制：
  - 重寫了 DataLoader 的讀取方式，新增了手動切換開關，讓訓練過程可以在「隨機讀取 (Unpaired / Random)」與「配對資料讀取 (Paired)」之間靈活調整。

- 訓練測試：
  - 在此配置下，已經成功完成了 50 Epochs 與 200 Epochs 的訓練與效果比較。<br/>
 
- Loss Curve 走勢圖：<br/>
  - Epochs 50：
![Epochs 50](/results/medical_paired/loss_convergence_smooth.png)
  - Epochs 200：
![Epochs 200](/results/medical_paired_200/loss_convergence_smooth.png)

- 數據比較：進行 Liang(2019) 論文數據與分別經過 50 Epochs 和 200 Epochs 的訓練後數據的比較。

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |
| Epoch_50 | 20.21 ± 2.95 | 89.18 ± 12.50 | 0.95 ± 0.01 | 33.28 ± 1.30 |
| Epoch_200 | 12.53 ± 3.25 | 69.76 ± 20.31 | 0.97 ± 0.01 | 35.65 ± 2.41 |
<br/>

- 實際圖片比較：<br/>
> 完整檔案為 /results 的 Paired.html 與 RunVSPaired.html<br/>

  - Paired L1 Loss 的訓練 Epochs 短長比較：
![Paired.png](/results/html_img/Paired.png)
  - 有無加入 Paired L1 Loss 的比較：
![MedicalVSPaired.png](/results/html_img/MedicalVSPaired.png)

 ## Phase 3
為了進一步解決深層特徵丟失與邊界模糊的問題，目前正在嘗試將 Generator 從 U-Net 升級為 UNet++ (Nested UNet)。

- 密集嵌套跳躍路徑：：
  - 引入 Nested Dense Skip Pathways，讓解碼器能接收來自編碼器同層及所有中間節點的整合訊號。

- 硬體資源突破：
  - 針對因為 UNet++ 海量特徵圖導致的 OOM (Out of Memory) 問題，成功導入 PyTorch 的梯度檢查點 (Gradient Checkpointing) 技術，以時間換取空間，大幅減少顯存佔用。

- 目前進度：
  - 已順利啟動訓練，並完成 50 Epochs 的初步效果驗證與測試。
  - 目前正在進行 200 Epochs 的訓練。<br/>

- Loss Curve 走勢圖：<br/>
  - Without Paired L1 Loss
    - Epochs 50：
![Epochs 50](/results/unet2plus_run/loss_convergence_smooth.png)
  - With Paired L1 Loss
    - Epochs 50：
![Epochs 50](/results/unet2plus_paired/loss_convergence_smooth.png)
    - Epochs 200：
![Epochs 200](/results/unet2plus_paired_200/loss_convergence_smooth.png)

-- 數據比較：進行 Liang(2019) 論文數據與有無加入Paired L1 Loss後經過 50 Epochs 訓練的數據比較。

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |
| (without) Epoch_50 | 21.18 ± 3.82 | 81.35 ± 15.56 | 0.95 ± 0.02 | 34.15 ± 1.72 |
| (with) Epoch_50 | 21.54 ± 4.42| 90.80296 ± 16.97 | 0.95 ± 0.01 | 33.19 ± 1.73 |
<br/>

- 實際圖片比較：<br/>
> 完整檔案為 /results 的 Unet++_RunVSPaired.html<br/>

![u2_MedicalVSPaired](/results/html_img/u2_MedicalVSPaired.png)

 ## Phase 4
在跨機構或不同設備來源的數據集中，CBCT 的 HU 值往往缺乏標準物理定義，且容易受到散射線影響，導致不同病患間出現強度不一致與數值漂移（Value Shift）的問題。例如空氣區域在某些病患的 CBCT 中數值可能高達數百，而非標準的 -1000。

- 背景強制鎖定與去噪
  - 善用數據集提供的 Patient Outline binary Mask，將背景空氣區域強制歸一化為 -1.0（對應 CycleGAN 的最暗像素值）。

- 基於百分位數的動態強度標準化
  - CBCT 影像：排除背景後，僅針對人體組織內部（Mask > 0）計算動態百分位數，消除極端值（如金屬偽影）後線性對齊至 [-1, 1] 區間。
  - CT 影像：維持標準 HU 範圍 [-1000,3000] 進行剪裁與歸一化，同樣強制將背景鎖定為 -1.0。

- 切片過濾
  - 改用 Mask 內人體組織像素佔比（>1%）作為過濾門檻，剔除無組織的純空氣切片。
 
- 雙格式支援
  - 支援將前處理後的數據直接輸出為 .npy 矩陣格式；同時優化了 DataLoader，使其能在訓練時動態兼容 .npy 與 .dcm 兩種輸入格式。 

- Loss Curve 走勢圖：<br/>
  - Without Paired L1 Loss
    - Epochs 50：
![Epochs 50](/results/1_medical_run/loss_convergence_smooth.png)
  - With Paired L1 Loss
    - Epochs 50：
![Epochs 50](/results/1_medical_paired/loss_convergence_smooth.png)

- 數據比較：進行 Liang(2019)論文數據、Phase 1 數據與進行前處理後的數據比較。Phase 1 數據與進行前處理後的數據都是進行 50 epochs 的比較。<br/>

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |
| Phase 1 | 24.27264 | 92.50987 | 0.94671 | 33.05643 |
<br/>

- 實際圖片比較：<br/>

# Engineering & Evaluation
在模型架構之外，本專案也針對測試流程與評估準確性進行了大幅度的工程優化：

- 評估指標 3D 體積化：
  - 根據 Linag (2019)論文中描寫的評估方式，將原本直接利用 2D 切片的計算方式重寫了評估腳本 (evaluate3D.py)。
  - 引入緩存機制，先將同一個病患的所有 2D 預測切片與真實切片沿著 Z 軸堆疊 (Stack) 成完整的 3D Numpy 陣列，再一次性進行計算。
  - 依照論文標準，精準設定 3D PSNR 與 3D SSIM 的訊號最大差異動態範圍為 4071。

- 訓練監控與 Loss 視覺化：
  - 撰寫了專屬的 Python 解析腳本，自動解析 loss_log.txt (plot_loss.py)。
  - 將龐雜的數值繪製成清晰的 Loss 折線圖，畫出「標準 GAN 誤差」與「Paired L1 誤差」。
  - 引入了移動平均 (Rolling Average) 技術，將劇烈震盪的單步損失轉換成平滑的收斂趨勢線，方便直觀評估訓練狀況。

- 背景排隊與自動偵測
  - 優化了訓練流程的自動化排程，處理在多 Epoch 長時間訓練下的資源管理。
  - 包含對顯示卡資源的背景殘留進程偵測（如利用 expandable_segments:True 解決記憶體碎片化），確保龐大的 UNet++ 訓練任務可以穩定在背景排隊與無縫執行。

# Environment Setup
本專案使用現代化的 Python 3.11 與 PyTorch 2.4.0 配置，支援 CUDA 12.1 硬體加速，並使用 Linux 伺服器。以下為完整的建置指南：
### 自行安裝 Conda (Miniconda 或 Anaconda)
> 注意：由於 GitHub 有單一檔案上傳的大小限制，本儲存庫無法直接附帶 Conda 的安裝檔。請務必先自行安裝 Conda 環境。 <br/>
強烈推薦安裝輕量級的 Miniconda。Linux 伺服器可以透過以下指令從終端機直接下載並安裝：
  1. 從官方伺服器下載 Miniconda Linux 安裝檔
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
  2. 賦予執行權限並執行安裝程式
```
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
  3. 重新載入終端機設定，讓 conda 指令立刻生效
```
source ~/.bashrc
```

### 執行 environment.yml 建立虛擬環境
請確保終端機目前位於本專案的根目錄下，且目錄中包含統整好的 environment.yml 檔案。接著執行以下指令，Conda 會自動根據檔案內容建立環境並下載所有必需的套件（包含 PyTorch、影像處理與 3D 評估等相依套件）：
```
conda env create -f environment.yml
```

### 啟動虛擬環境
環境安裝完成後，請執行以下指令啟動虛擬環境：
```
conda activate img2img
```
