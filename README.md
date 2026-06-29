# pytorch_CycleGAN
本專案基於原始 PyTorch CycleGAN 框架進行深度改動，旨在實現精準的醫學影像翻譯（CBCT-to-CT）。專案的初期架構靈感與參數設定參考自醫學期刊論文 Liang 等人(2019)，並在後續階段針對配對資料集的優勢進行了 Loss Function 的改寫，目前正積極導入更深層的 UNet++ 架構以期獲得更極致的解剖結構保留效果。

原始 CycleGAN 架構：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
出自論文：https://doi.org/10.48550/arXiv.1703.10593 and https://doi.org/10.48550/arXiv.1611.07004
初期架構參考論文：https://doi.org/10.48550/arXiv.1810.13350

# Project Stages
