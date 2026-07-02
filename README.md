# pytorch_CycleGAN
本專案基於原始 PyTorch CycleGAN 框架進行深度改動，旨在實現精準的醫學影像翻譯（CBCT-to-CT）。專案的初期架構靈感與參數設定參考自醫學期刊論文 Liang 等人(2019)，並在後續階段針對配對資料集的優勢進行了 Loss Function 的改寫，目前正積極導入更深層的 UNet++ 架構以期獲得更極致的解剖結構保留效果。

> 原始 CycleGAN 架構：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git <br/>
> 出自論文：https://doi.org/10.48550/arXiv.1703.10593 和 https://doi.org/10.48550/arXiv.1611.07004 <br/>
> 初期架構參考論文：https://doi.org/10.48550/arXiv.1810.13350

# Project Stages
本專案的演進主要分為以下三個階段：

- Phase 1: 對齊 Liang (2019) 論文架構
- Phase 2: 活用配對資料集優勢
- Phase 3: 生成器升級為 UNet++ (進行中)

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

- 數據比較：<br/>

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |
| Epoch_50 | 24.27264 | 92.50987 | 0.94671 | 33.05643 |
| Epoch_200 | 14.69166 | 75.65056 | 0.960477 | 34.90914 |

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

- 數據比較：<br/>

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |
| Epoch_50 | 20.21277 | 89.17818 | 0.951166 | 33.27695 |
| Epoch_200 | 12.53239 | 69.76138 | 0.97451 | 35.65066 |

 ## Phase 3
為了進一步解決深層特徵丟失與邊界模糊的問題，目前正在嘗試將 Generator 從 U-Net 升級為 UNet++ (Nested UNet)。

- 密集嵌套跳躍路徑：：
  - 引入 Nested Dense Skip Pathways，讓解碼器能接收來自編碼器同層及所有中間節點的整合訊號。

- 硬體資源突破：
  - 針對因為 UNet++ 海量特徵圖導致的 OOM (Out of Memory) 問題，成功導入 PyTorch 的梯度檢查點 (Gradient Checkpointing) 技術，以時間換取空間，大幅減少顯存佔用。

- 目前進度：
  - 已順利啟動訓練，目前正在進行 50 Epochs 的初步效果驗證與測試。<br/>
- Loss Curve 走勢圖：<br/>
  - Epochs 50 (without Paired L1 Loss)：
![Epochs 50](/results/unet2plus_run/loss_convergence_smooth.png)
  - Epochs 50 (with Paired L1 Loss)：
![Epochs 50](/results/unet2plus_paired/loss_convergence_smooth.png)

- 數據比較：<br/>

| 比較項目 | MAE | RMSE | SSIM | PSNR |
| --- | --- | --- | --- | --- |
| Liang(2019)論文數據 | 29.85 ± 4.94 | 84.46 ± 12.40 | 0.85 ± 0.03 | 30.65 ± 1.36 |


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
