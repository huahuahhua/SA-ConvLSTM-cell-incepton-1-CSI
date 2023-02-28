import torch
import torch.nn as nn


class self_attention_memory_module(nn.Module):  # SAM
    def __init__(self, input_dim, hidden_dim):
        super(self_attention_memory_module, self).__init__()
        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W
        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        # memory m attention #
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        # 논문의 수식과 같습니다(figure)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class inception(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super(inception, self).__init__()
        self.input_channels = in_channel
        self.hidden_dim = hidden_dim

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(self.hidden_dim, self.hidden_dim))  # (num_groups, num_channels)

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(self.hidden_dim, self.hidden_dim),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(self.hidden_dim, self.hidden_dim))  # (num_groups, num_channels)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dim, kernel_size=1, padding=0),
            nn.GroupNorm(self.hidden_dim, self.hidden_dim))  # (num_groups, num_channels)

        self.out = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, kernel_size=1, padding=0),
            nn.GroupNorm(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        x_1x1 = self.conv1x1(x)
        x_3x3 = self.conv3x3(x)
        x_3x3_2 = self.conv3x3_2(x)

        x_cat = torch.cat([x_1x1, x_3x3, x_3x3_2], dim=1)

        out = self.out(x_cat)
        return out


class SA_Convlstm_cell(nn.Module):
    def __init__(self, in_channel, hidden_dim, configs):
        super(SA_Convlstm_cell, self).__init__()
        # hyperparrams
        self.input_channels = in_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = configs.filter_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.device = configs.device

        self.attention_layer = self_attention_memory_module(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.i = inception(in_channel=in_channel + self.hidden_dim, hidden_dim=hidden_dim)
        self.f = inception(in_channel=in_channel + self.hidden_dim, hidden_dim=hidden_dim)
        self.o = inception(in_channel=in_channel + self.hidden_dim, hidden_dim=hidden_dim)
        self.g = inception(in_channel=in_channel + self.hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x, hidden):
        h, c, m = hidden
        combined = torch.cat([x, h], dim=1)  # combine 해서 (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])
        i = self.i(combined)
        f = self.i(combined)
        o = self.i(combined)
        g = self.i(combined)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)  # 第一个用来外面的norm_layer

    def init_hidden(self, batch_size, img_size):  # h, c, m initalize
        h, w = img_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))


class con5x5(nn.Module):
    def __init__(self):
        super(con5x5, self).__init__()
        self.convd = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.GroupNorm(64, 64)
        )


if __name__ == '__main__':
    from configs.radar_train_configs import configs

    #
    parse = configs()
    configs = parse.parse_args()
    model = SA_Convlstm_cell(configs.num_hidden, configs.num_hidden, configs).cuda()

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
