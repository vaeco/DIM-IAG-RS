import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 空间一致误差
# 不希望推理后的图像与原来相比，
# 某像素的值和其相邻像素的值的差发生过大的改变
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()

        kernel_left = torch.FloatTensor([[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]).to(device).unsqueeze(0)
        kernel_right = torch.FloatTensor([[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]).to(device).unsqueeze(0)
        kernel_up = torch.FloatTensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]).to(device).unsqueeze(0)
        kernel_down = torch.FloatTensor([[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]).to(device).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

        self.pool = nn.AvgPool2d(4)

    def calculate_E(self, org_pool, enhance_pool):
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return torch.mean(E)

    def forward(self, org, enhance):
        org_R, org_G, org_B = torch.split(org, 1, dim=1)
        enhance_R, enhance_G, enhance_B = torch.split(enhance, 1, dim=1)

        org_R_pool = self.pool(org_R)
        org_G_pool = self.pool(org_G)
        org_B_pool = self.pool(org_B)
        enhance_R_pool = self.pool(enhance_R)
        enhance_G_pool = self.pool(enhance_G)
        enhance_B_pool = self.pool(enhance_B)

        E_R = self.calculate_E(org_R_pool, enhance_R_pool)
        E_G = self.calculate_E(org_G_pool, enhance_G_pool)
        E_B = self.calculate_E(org_B_pool, enhance_B_pool)
        E = (E_R + E_G + E_B)

        return E


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class L_exp(nn.Module):

    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(device), 2))
        return d

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k


import torch
import torch.nn as nn


class HistogramLoss(nn.Module):
    def __init__(self, bins=256, range=(0, 1), norm=True, reduction='mean'):
        """
        初始化直方图损失函数

        参数:
            bins: 直方图的 bin 数目
            range: 直方图的范围
            norm: 是否对直方图进行归一化
            reduction: 选择如何对输出进行聚合，可选 'none', 'mean', 'sum'
        """
        super(HistogramLoss, self).__init__()
        self.bins = bins
        self.range = range
        self.norm = norm
        self.reduction = reduction

    def forward(self, img1, img2):
        """
        计算两张图片之间的直方图损失

        参数:
            img1: 第一张图片 (Tensor)
            img2: 第二张图片 (Tensor)

        返回:
            直方图损失
        """
        # 计算第一张图片的直方图
        hist1 = torch.histc(img1, bins=self.bins, min=self.range[0], max=self.range[1])

        # 计算第二张图片的直方图
        hist2 = torch.histc(img2, bins=self.bins, min=self.range[0], max=self.range[1])

        # 对直方图进行归一化（如果需要）
        if self.norm:
            hist1 = hist1 / torch.sum(hist1)
            hist2 = hist2 / torch.sum(hist2)

        # 计算直方图的 L1 距离（曼哈顿距离）
        loss = torch.sum(torch.abs(hist1 - hist2))

        # 应用 reduction
        if self.reduction == 'mean':
            loss = loss / self.bins
        elif self.reduction == 'sum':
            pass  # L1 距离已经在 sum
        elif self.reduction == 'none':
            loss = torch.abs(hist1 - hist2)

        return loss


if __name__ == '__main__':
    hist_loss = HistogramLoss(norm=True,reduction='sum')
    x=torch.rand(1,3,256,256).to(0)
    y=torch.rand(1,3,256,256).to(0)
    print(hist_loss(x,y))
    L1Loss = nn.SmoothL1Loss()
    print(L1Loss(x,y))

