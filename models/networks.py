import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.utils.checkpoint import checkpoint


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_128 | unet_256
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "liang_unet":
        net = LiangGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG == 'liang_unet2plus':
        net = LiangUnet2PlusGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class LiangGenerator(nn.Module):
    """基於 Liang(2019) 論文實作的醫療影像 U-Net"""
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(LiangGenerator, self).__init__()
        
        # Helper: Conv + Norm + LeakyReLU
        def conv_block(in_c, out_c, stride=1, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=True),
                norm_layer(out_c),
                nn.LeakyReLU(0.2, True)
            )

        # Helper: Upsample + Conv + Norm + LeakyReLU
        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_c),
                nn.LeakyReLU(0.2, True)
            )

        # Encoder (downsampling path)
        # 512 -> 256
        self.down1 = nn.Sequential(conv_block(input_nc, ngf, 1), conv_block(ngf, ngf, 2))
        # 256 -> 128
        self.down2 = nn.Sequential(conv_block(ngf, ngf*2, 1), conv_block(ngf*2, ngf*2, 2))
        # 128 -> 64
        self.down3 = nn.Sequential(conv_block(ngf*2, ngf*4, 1), conv_block(ngf*4, ngf*4, 2))
        # 64 -> 32
        self.down4 = nn.Sequential(conv_block(ngf*4, ngf*8, 1), conv_block(ngf*8, ngf*8, 2))
        # 32 -> 16
        self.down5 = nn.Sequential(conv_block(ngf*8, 512, 1), conv_block(512, 512, 2))

        # Bottleneck
        # 16x16, 1x1 Convolution
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        )

        # Decoder (Upsampling path)
        # 16 -> 32 (Concat 512 from down5)
        self.up1 = up_block(512 + 512, 512)
        # 32 -> 64 (Concat ngf*8 from down4)
        self.up2 = up_block(512 + ngf*8, ngf*8)
        # 64 -> 128 (Concat ngf*4 from down3)
        self.up3 = up_block(ngf*8 + ngf*4, ngf*4)
        # 128 -> 256 (Concat ngf*2 from down2)
        self.up4 = up_block(ngf*4 + ngf*2, ngf*2)
        # 256 -> 512 (Concat ngf from down1)
        self.up5 = up_block(ngf*2 + ngf, ngf)

        # Final Output Layer
        self.final = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder & Save skips
        d1 = self.down1(x)   # 256
        d2 = self.down2(d1)  # 128
        d3 = self.down3(d2)  # 64
        d4 = self.down4(d3)  # 32
        d5 = self.down5(d4)  # 16
        
        # Center
        bn = self.bottleneck(d5)
        
        # Decoder with Skip Connections (Concatenation)
        u1 = self.up1(torch.cat([bn, d5], 1))
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))
        u5 = self.up5(torch.cat([u4, d1], 1))
        
        return self.final(u5)
        
class LiangUnet2PlusGenerator(nn.Module):
    """基於 Liang(2019) 結構擴展之醫療影像 UNet++ (Nested UNet)"""
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(LiangUnet2PlusGenerator, self).__init__()
        
        # 遵循 Liang 論文的 5 層特徵通道數設定 (5層降採樣加上最後的第0層)
        # 對應的特徵維度矩陣中的 i 索引 (0 到 5)
        self.filters = [ngf, ngf, ngf*2, ngf*4, ngf*8, 512] 
        self.num_downs = 5  # 5次下採樣

        # 1. 輔助區塊定義
        def conv_block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=True),
                norm_layer(out_c),
                nn.LeakyReLU(0.2, True)
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_c),
                nn.LeakyReLU(0.2, True)
            )

        # 2. 建立 UNet++ 的密集網格模組
        self.blocks = nn.ModuleDict()
        self.upsamples = nn.ModuleDict()

        for i in range(self.num_downs + 1):
            for j in range(self.num_downs + 1 - i):
                node_name = f"x_{i}_{j}"
                
                if j == 0:
                    # ---- Encoder 骨幹 (j = 0) ----
                    if i == 0:
                        # 頂層輸入節點 (512x512)
                        self.blocks[node_name] = conv_block(input_nc, self.filters[i], stride=1)
                    else:
                        # 降採樣節點：結合 1x1 conv 與 2x2 跨步卷積 (遵循 Liang 原版設計)
                        in_c = self.filters[i-1]
                        out_c = self.filters[i]
                        self.blocks[node_name] = nn.Sequential(
                            conv_block(in_c, out_c, stride=1),
                            conv_block(out_c, out_c, stride=2)
                        )
                else:
                    # ---- UNet++ 密集嵌套與解碼器節點 (j > 0) ----
                    # 計算 Concatenation 後輸入進來的所有通道加總：
                    # 同層前面有 j 個節點，加上下方節點上採樣後送上來的特徵圖
                    
                    # 1. 計算同層前面所有節點 (X_i,0 到 X_i,j-1) 的通道總和
                    skip_channels = sum([self.filters[i] for _ in range(j)])
                    
                    # 2. 計算下方節點 (X_i+1, j-1) 經過 up_block 之後的輸出通道數
                    # 為了符合結構對齊，上採樣後的通道數與目前水平層的 filters[i] 相同
                    up_channels = self.filters[i] 
                    
                    in_c = skip_channels + up_channels
                    out_c = self.filters[i]
                    
                    # 定義下方節點往上送的上採樣層
                    up_name = f"up_{i+1}_{j-1}"
                    self.upsamples[up_name] = up_block(self.filters[i+1], out_c)
                    
                    # 節點卷積融合
                    self.blocks[node_name] = conv_block(in_c, out_c, stride=1)

        # 3. 論文最後的 Bottleneck (1x1 Convolution 作用在最底層 X_5,0 之後)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        )

        # 4. Final Output Layer (從最高層最右端 X_0,5 輸出)
        self.final = nn.Sequential(
            nn.Conv2d(self.filters[0], output_nc, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 用於儲存所有二維網格節點輸出的字典
        outputs = {}

        # 依序計算網格 (以行 j 為主軸，由左至右、由下至上計算)
        for j in range(self.num_downs + 1):
            for i in range(self.num_downs + 1 - j):
                node_name = f"x_{i}_{j}"
                
                if j == 0:
                    # Encoder 傳導
                    if i == 0:
                        outputs[node_name] = self.blocks[node_name](x)
                    else:
                        outputs[node_name] = self.blocks[node_name](outputs[f"x_{i-1}_{j}"])
                        
                    # 當走到最底層的 Encoder 節點 (x_5_0) 時，過一下 1x1 Bottleneck
                    if i == self.num_downs and j == 0:
                        outputs[node_name] = self.bottleneck(outputs[node_name])
                else:
                    """慢"""
                    # 1. 先在外部拼接好特徵圖 (需要保留拼接結果傳給 checkpoint)
                    up_feat = self.upsamples[f"up_{i+1}_{j-1}"](outputs[f"x_{i+1}_{j-1}"])
                    skip_feats = [outputs[f"x_{i}_{k}"] for k in range(j)]
                    cat_feat = torch.cat(skip_feats + [up_feat], dim=1)
                    
                    # 2. 如果模型在訓練狀態，且該張量需要梯度，就調用 checkpoint
                    if self.training and cat_feat.requires_grad:
                        # checkpoint 內部會暫時關閉計算圖，直到 backward 時才重新前向計算該 block
                        outputs[node_name] = checkpoint(self.blocks[node_name], cat_feat, use_reentrant=False)
                    else:
                        outputs[node_name] = self.blocks[node_name](cat_feat)
                    
                    
                    """快
                    # 嵌套及解碼器傳導
                    # 1. 取得下方節點並進行上採樣
                    up_feat = self.upsamples[f"up_{i+1}_{j-1}"](outputs[f"x_{i+1}_{j-1}"])
                    
                    # 2. 收集同層前面所有的 Skip Connection 特徵
                    skip_feats = [outputs[f"x_{i}_{k}"] for k in range(j)]
                    
                    # 3. 拼接所有特徵圖
                    cat_feat = torch.cat(skip_feats + [up_feat], dim=1)
                    
                    # 4. 通過融合卷積層
                    outputs[node_name] = self.blocks[node_name](cat_feat)
                    """
        # 從最高、最右側的節點 (x_0_5) 導出最終醫療影像結果
        final_node = f"x_0_{self.num_downs}"
        return self.final(outputs[final_node])