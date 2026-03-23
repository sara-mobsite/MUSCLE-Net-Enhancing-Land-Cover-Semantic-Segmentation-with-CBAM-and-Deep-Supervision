# MUSCLE-Net-Enhancing-Land-Cover-Semantic-Segmentation-with-CBAM-and-Deep-Supervision
Official implementation of MUSCLE-Net, a land-cover semantic segmentation model with CBAM and deep supervision for DFC2020 and DynamicEarthNet
## MUSCLE-Net architecture

The figure below illustrates the overall architecture of MUSCLE-Net, including the encoder, CBAM-enhanced decoder, and deep supervision branches.

![MUSCLE-Net architecture](images/systemAll.jpg)



### 🌱 Efficiency and Environmental Impact

| Model | Time / Epoch (s) | Avg. Epochs (5 runs) | Emissions (kg CO₂eq / epoch) |
|-------|-------------------|----------------------|-------------------------------|
| **MUSCLE-Net** | 296.57 | 86.8 ± 30.15 | 0.000072 |
| UNet | 64.38 | 107.8 ± 53.80 | 0.000017 |
| DeepLabV3 | 217.69 | 73.4 ± 30.07 | 0.000053 |
| PSPNet | 45.00 | 75.6 ± 25.77 | 0.000012 |

