# MUSCLE-Net-Enhancing-Land-Cover-Semantic-Segmentation-with-CBAM-and-Deep-Supervision
Official implementation of MUSCLE-Net, a land-cover semantic segmentation model with CBAM and deep supervision for DFC2020 and DynamicEarthNet
## MUSCLE-Net architecture

The figure below illustrates the overall architecture of MUSCLE-Net, including the encoder, CBAM-enhanced decoder, and deep supervision branches.

![MUSCLE-Net architecture](images/systemAll.jpg)





## 📊 Quantitative Results

### Average IoU per Run

| Run | UNet  | PSPNet | DeepLabV3 | **MUSCLE-Net (ours)** |
|-----|-------|--------|-----------|------------------------|
| 0   | 0.5021 | 0.4514 | 0.4322 | **0.5324** |
| 1   | 0.5248 | 0.4793 | 0.4294 | **0.5322** |
| 2   | 0.5104 | 0.4695 | 0.4646 | **0.5326** |
| 3   | 0.4941 | 0.4913 | 0.4619 | **0.5080** |
| 4   | 0.5208 | 0.4292 | 0.4448 | **0.5460** |
| 5   | 0.5342 | 0.4467 | 0.4621 | **0.5476** |
| 6   | 0.5097 | 0.4884 | 0.4446 | **0.5297** |
| 7   | 0.5242 | 0.4704 | 0.4575 | **0.5353** |
| 8   | 0.4939 | 0.4588 | 0.4659 | **0.5334** |
| 9   | 0.4875 | 0.4791 | 0.4684 | **0.5373** |

### Accuracy per Run

| Run | UNet  | PSPNet | DeepLabV3 | **MUSCLE-Net (ours)** |
|-----|-------|--------|-----------|------------------------|
| 0   | 0.6547 | 0.6297 | 0.6322 | **0.6905** |
| 1   | 0.7020 | 0.6643 | 0.6157 | **0.7098** |
| 2   | 0.6810 | 0.6570 | 0.6519 | **0.7042** |
| 3   | 0.6498 | 0.6780 | 0.6489 | **0.7040** |
| 4   | 0.6745 | 0.6270 | 0.6238 | **0.7027** |
| 5   | 0.6983 | 0.6288 | 0.6400 | **0.7157** |
| 6   | 0.6638 | 0.6626 | 0.6334 | **0.6943** |
| 7   | 0.6830 | 0.6555 | 0.6588 | **0.6948** |
| 8   | 0.6509 | 0.6670 | 0.6472 | **0.6997** |
| 9   | 0.6519 | 0.6627 | 0.6609 | **0.6990** |

### 🌱 Efficiency and Environmental Impact

| Model | Time / Epoch (s) | Avg. Epochs (5 runs) | Emissions (kg CO₂eq / epoch) |
|-------|-------------------|----------------------|-------------------------------|
| **MUSCLE-Net** | 296.57 | 86.8 ± 30.15 | 0.000072 |
| UNet | 64.38 | 107.8 ± 53.80 | 0.000017 |
| DeepLabV3 | 217.69 | 73.4 ± 30.07 | 0.000053 |
| PSPNet | 45.00 | 75.6 ± 25.77 | 0.000012 |

