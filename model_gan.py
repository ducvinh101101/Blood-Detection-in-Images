import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Block tích hợp để duy trì thông tin của ảnh gốc trong quá trình biến đổi."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Kết hợp đầu vào với output để giữ thông tin gốc

class Generator(nn.Module):
    """Bộ tạo ảnh chuyển đổi từ NoBlood sang Blood."""
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        # Lớp đầu vào
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling (Giảm kích thước)
        self.down_blocks = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual Blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_residual_blocks)])

        # Upsampling (Tăng kích thước)
        self.up_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Lớp đầu ra
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        return self.final(x)
class Discriminator(nn.Module):
    """Bộ phân biệt dùng để xác định ảnh thật hay giả."""
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, stride):
            """Tạo một lớp convolution block để giảm kích thước ảnh và phân tích đặc trưng."""
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            discriminator_block(input_channels, 64, stride=2),
            discriminator_block(64, 128, stride=2),
            nn.InstanceNorm2d(128),
            discriminator_block(128, 256, stride=2),
            nn.InstanceNorm2d(256),
            discriminator_block(256, 512, stride=1),
            nn.InstanceNorm2d(512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Đầu ra 1 node (probability real/fake)
        )

    def forward(self, x):
        return self.model(x)
