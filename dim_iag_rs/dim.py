from .conv import *


class Encoder(nn.Module):
    def __init__(self, c1=3, c_hidden=32):
        super().__init__()

        self.conv1 = Conv(c1=c1, c2=c_hidden, k=3, s=2)
        self.conv2 = Conv(c1=c_hidden, c2=c_hidden * 2, k=3, s=2)
        self.conv3 = Conv(c1=c_hidden * 2, c2=c_hidden * 4, k=3, s=2)

        self.stage1 = C3k2(c1=c_hidden, c2=c_hidden, c3k=False, e=0.25)
        self.stage2 = C3k2(c1=c_hidden * 2, c2=c_hidden * 2, c3k=False, e=0.25)

        self.attn1 = A2C2f(c1=c_hidden * 4, c2=c_hidden * 4, area=1, residual=True, e=0.25)

    def forward(self, x):
        encoder_1 = self.stage1(self.conv1(x))
        encoder_2 = self.stage2(self.conv2(encoder_1))

        attn = self.attn1(self.conv3(encoder_2))

        return encoder_1, encoder_2, attn


class DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=32):
        super().__init__()
        self.encoder = Encoder(c1=c1, c_hidden=c_hidden)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0 = Conv(c1=c_hidden * 4, c2=c_hidden * 2)
        self.conv1 = Conv(c1=c_hidden * 4, c2=c_hidden)
        self.conv2 = Conv(c1=c_hidden * 2, c2=c_hidden)
        self.decoder = Conv(c1=c_hidden, c2=3, k=3)

    def forward(self, x):
        encoder_1, encoder_2, attn = self.encoder(x)

        attn = self.upsample(self.conv0(attn))
        fusion_1 = torch.cat([attn, encoder_2], 1)
        IM_1 = self.intensity_mapping(fusion_1)
        IM_1 = self.conv1(IM_1)
        IM_1 = self.upsample(IM_1)

        fusion_2 = torch.cat([IM_1, encoder_1], 1)
        IM_2 = self.intensity_mapping(fusion_2)
        IM_2 = self.conv2(IM_2)
        IM_2 = self.upsample(IM_2)

        output = self.decoder(IM_2)

        return output

    @staticmethod
    def intensity_mapping(x):
        return x * 2 - torch.pow(x, 2)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(1, 3, 800, 1200).to(device)
    model = DIM().to(device)
    output = model(input_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params, output.size())
