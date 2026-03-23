import torch
from torch import nn
import torch.nn.functional as F
from models.attention import CBAM


class MUSCLENet(nn.Module):
    """
    MUSCLE-Net with:
    - encoder backbone
    - CBAM-enhanced decoder
    - one auxiliary branch at H/4 and W/4
    - best auxiliary loss weight = 0.10
    """
    def __init__(self, original_model, num_classes: int = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            original_model.model.vision_encoder.conv1,
            original_model.model.vision_encoder.bn1,
            original_model.model.vision_encoder.act1,
            original_model.model.vision_encoder.maxpool,
            original_model.model.vision_encoder.layer1,
            original_model.model.vision_encoder.layer2,
            original_model.model.vision_encoder.layer3,
            original_model.model.vision_encoder.layer4,
        )

        for param in self.encoder.parameters():
            param.requires_grad = True

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.cbam1 = CBAM(512)

        # Auxiliary branch produced at H/4, W/4
        self.auxiliary_conv11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.cbam_auxiliary11 = CBAM(256)

        self.auxiliary_conv21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cbam_auxiliary21 = CBAM(128)

        self.auxiliary_conv211 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.cbam_auxiliary211 = CBAM(64)

        self.auxiliary_conv111 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.conv2 = nn.Conv2d(512 + 64 + 64, 256, kernel_size=3, padding=1)
        self.cbam2 = CBAM(256)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cbam3 = CBAM(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.cbam4 = CBAM(64)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.cbam5 = CBAM(32)

        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.cbam6 = CBAM(16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2], x.shape[3]

        # Early skip feature
        x_skip = self.encoder[0:3](x)

        # Deep encoder output
        x_enc = self.encoder(x)

        # Decoder starts at H/4, W/4
        x = F.interpolate(x_enc, size=(h // 4, w // 4), mode="bilinear", align_corners=False)
        x = F.relu(self.conv1(x))
        x = self.cbam1(x)

        # Auxiliary branch at H/4, W/4
        aux = F.relu(self.auxiliary_conv11(x))
        aux = self.cbam_auxiliary11(aux)

        aux = F.relu(self.auxiliary_conv21(aux))
        aux = self.cbam_auxiliary21(aux)

        aux = F.relu(self.auxiliary_conv211(aux))
        aux_skip = self.cbam_auxiliary211(aux)

        auxiliary_output = self.auxiliary_conv111(aux_skip)

        # Merge main branch + early skip + auxiliary features
        x = F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
        aux_skip_up = F.interpolate(aux_skip, size=(h // 2, w // 2), mode="bilinear", align_corners=False)

        x = torch.cat([x, x_skip, aux_skip_up], dim=1)
        x = F.relu(self.conv2(x))
        x = self.cbam2(x)

        x = F.relu(self.conv3(x))
        x = self.cbam3(x)

        x = F.relu(self.conv4(x))
        x = self.cbam4(x)

        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = F.relu(self.conv5(x))
        x = self.cbam5(x)

        x = F.relu(self.conv6(x))
        x = self.cbam6(x)

        main_output = self.final_conv(x)

        return main_output, auxiliary_output
