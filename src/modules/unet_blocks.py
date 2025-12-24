import torch
from torch import nn
from torchvision.ops import DeformConv2d

from .attention import (SpatialTransformer, 
                        OffsetRefStrucInter, 
                        ChannelAttnBlock,
                        SelfAttentionBlock,
                        WindowAttentionBlock,
                        ContentGuidedGate)
from .resnet import (Downsample2D, 
                     ResnetBlock2D, 
                     Upsample2D)


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    channel_attn=False,
    content_channel=32,
    reduction=32):

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding)
    elif down_block_type == "MCADownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return MCADownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            channel_attn=channel_attn,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel,
            reduction=reduction)
    else:
        raise ValueError(f"{down_block_type} does not exist.")

def get_down_block_with_parallel(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    channel_attn=False,
    content_channel=32,
    reduction=32,
    window_size=4):

    down_block_type = down_block_type[7:] if down_block_type. startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding)
    elif down_block_type == "MCADownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return MCADownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            channel_attn=channel_attn,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel,
            reduction=reduction)
    elif down_block_type == "ParallelDownBlock2D":
        return ParallelDownBlock2D(
            num_layers=1,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=False,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            window_size=window_size,
            content_channel=content_channel)
    else:
        raise ValueError(f"{down_block_type} does not exist.")
    
def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    upblock_index,
    resnet_groups=None,
    cross_attention_dim=None,
    structure_feature_begin=64):

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups)
    elif up_block_type == "StyleRSIUpBlock2D":
        return StyleRSIUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            structure_feature_begin=structure_feature_begin,
            upblock_index=upblock_index)
    else:
        raise ValueError(f"{up_block_type} does not exist.")

def get_up_block_with_parallel(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    upblock_index,
    resnet_groups=None,
    cross_attention_dim=None,
    structure_feature_begin=64,
    content_channel=32,
    window_size=4):

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups)
    elif up_block_type == "StyleRSIUpBlock2D":
        return StyleRSIUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            structure_feature_begin=structure_feature_begin,
            upblock_index=upblock_index)
    elif up_block_type == "ParallelUpBlock2D":
        return ParallelUpBlock2D(
            num_layers=1,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attn_num_head_channels=attn_num_head_channels,
            window_size=4,
            content_channel=content_channel)
    else:
        raise ValueError(f"{up_block_type} does not exist.")

class UNetMidMCABlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        channel_attn: bool = False,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        content_channel=256,
        reduction=32,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        content_attentions = []
        style_attentions = []

        for _ in range(num_layers):
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels + content_channel,
                    out_channels=in_channels,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            style_attentions.append(
                SpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.content_attentions = nn.ModuleList(content_attentions)
        self.style_attentions = nn.ModuleList(style_attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, 
        hidden_states, 
        temb=None, 
        encoder_hidden_states=None,
        index=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        for content_attn, style_attn, resnet in zip(self.content_attentions, self.style_attentions, self.resnets[1:]):
            
            # content
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)
            
            # t_embed
            hidden_states = resnet(hidden_states, temb)

            # style
            current_style_feature = encoder_hidden_states[0]
            batch_size, channel, height, width = current_style_feature.shape
            current_style_feature = current_style_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
            hidden_states = style_attn(hidden_states, context=current_style_feature)

        return hidden_states


class MCADownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        channel_attn: bool = False,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        content_channel=16,
        reduction=32,
    ):
        super().__init__()
        content_attentions = []
        resnets = []
        style_attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels+content_channel,
                    out_channels=in_channels,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            print("The style_attention cross attention dim in Down Block {} layer is {}".format(i+1, cross_attention_dim))
            style_attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
        self.content_attentions = nn.ModuleList(content_attentions)
        self.style_attentions = nn.ModuleList(style_attentions)
        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self, 
        hidden_states, 
        index,
        temb=None, 
        encoder_hidden_states=None
    ):
        output_states = ()

        for content_attn, resnet, style_attn in zip(self.content_attentions, self.resnets, self.style_attentions):
            
            # content
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)
            
            # t_embed
            hidden_states = resnet(hidden_states, temb)

            # style
            current_style_feature = encoder_hidden_states[0]
            batch_size, channel, height, width = current_style_feature.shape
            current_style_feature = current_style_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
            hidden_states = style_attn(hidden_states, context=current_style_feature)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class StyleRSIUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        structure_feature_begin=64, 
        upblock_index=1,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        sc_interpreter_offsets = []
        dcn_deforms = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.upblock_index = upblock_index

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            
            sc_interpreter_offsets.append(
                OffsetRefStrucInter(
                    res_in_channels=res_skip_channels,
                    # style_feat_in_channels=int(structure_feature_begin * 2 / upblock_index),
                    style_feat_in_channels=int(structure_feature_begin * (2 ** ( 3-self.upblock_index) )),
                    n_heads=attn_num_head_channels,
                    num_groups=resnet_groups,
                )
            )
            dcn_deforms.append(
                DeformConv2d(
                    in_channels=res_skip_channels,
                    out_channels=res_skip_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    dilation=1,
                )
            )

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
        self.sc_interpreter_offsets = nn.ModuleList(sc_interpreter_offsets)
        self.dcn_deforms = nn.ModuleList(dcn_deforms)
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.num_layers = num_layers

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        style_structure_features,
        index,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        
    ):
        total_offset = 0

        # style_content_feat = style_structure_features[-self.upblock_index-2]

        #TODO 临时修改，不精确
        style_content_feat = style_structure_features[-index-2]
        # print(f"style_content_feat: {style_content_feat.shape}")

        # ✅ 在循环外统一插值
        if style_content_feat.shape[2:] != res_hidden_states_tuple[-1].shape[2:]:
            import torch.nn.functional as F
            style_content_feat = F. interpolate(
                style_content_feat,
                size=res_hidden_states_tuple[-1].shape[2:],
                mode='bilinear',
                align_corners=False
            )

        for i, (sc_inter_offset, dcn_deform, resnet, attn) in \
            enumerate(zip(self.sc_interpreter_offsets, self.dcn_deforms, self.resnets, self.attentions)):
            # pop res hidden states 
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # Skip Style Content Interpreter by DCN
            offset = sc_inter_offset(res_hidden_states, style_content_feat)
            offset = offset.contiguous()
            # offset sum
            offset_sum = torch.mean(torch.abs(offset))
            total_offset += offset_sum

            res_hidden_states = res_hidden_states.contiguous()
            res_hidden_states = dcn_deform(res_hidden_states, offset)
            # concat as input
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        offset_out = total_offset / self.num_layers    

        return hidden_states, offset_out


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

# 重新设计的ParallelDownBlock2D - 类似MCADownBlock2D的结构
class ParallelDownBlock2D(nn.Module):
    """
    平行下采样块：在相同分辨率上进行特征精炼
    设计类似MCADownBlock2D，但使用自注意力而非跨注意力
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels: int = 1,
        attention_type: str = "default",
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = False,  # 平行层默认不下采样
        window_size: int = 4,  # ← 新增:  窗口注意力尺寸
        content_channel: int = 0,  # ← 新增:  内容通道数
    ):
        super().__init__()
        
        # 平行处理要求输入输出通道数相同
        assert in_channels == out_channels, f"Parallel layer requires same in/out channels, got {in_channels} -> {out_channels}"
        
        self.attention_type = attention_type
        self. attn_num_head_channels = attn_num_head_channels
        
        resnets = []
        local_attentions = []   # ← 新增: 窗口注意力
        global_attentions = []  # ← 保留:  全局注意力
        content_gates = []      # ← 新增: 内容引导门控

        for i in range(num_layers):
            # ResNet块：特征精炼
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,  # 保持通道数不变
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            
            # ← 新增: 局部窗口注意力 (保护细小笔画)
            local_attentions.append(
                WindowAttentionBlock(
                    channels=in_channels,
                    window_size=window_size,
                    num_heads=4,
                    num_groups=resnet_groups
                )
            )
            
            #全局自注意力 (保持原有)
            global_attentions. append(
                SelfAttentionBlock(
                    in_channels=in_channels,
                    n_heads=attn_num_head_channels,
                    d_head=in_channels // attn_num_head_channels,
                    dropout=dropout,
                    num_groups=resnet_groups,
                )
            )
            # global_attentions.append(
            #     WindowAttentionBlock(
            #         channels=in_channels,
            #         window_size=window_size*2,  # 扩大窗口覆盖全局
            #         num_heads=2,
            #         num_groups=resnet_groups
            #     )
            # )
            
            # ← 新增: 内容引导门控
            if content_channel > 0:
                content_gates.append(
                    ContentGuidedGate(
                        in_channels=in_channels,
                        content_channels=content_channel,
                        num_groups=resnet_groups
                    )
                )
            else:
                content_gates.append(None)
        
        self.resnets = nn.ModuleList(resnets)
        self.local_attentions = nn.ModuleList(local_attentions)
        self.global_attentions = nn.ModuleList(global_attentions)
        self.content_gates = nn.ModuleList(content_gates)
        # 平行层通常不需要下采样，但保持接口一致性
        if add_downsample:
            print("Warning: ParallelDownBlock2D with add_downsample=True may break parallel processing")
            self.downsamplers = nn.ModuleList([
                Downsample2D(
                    out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                )
            ])
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        print(f"Created ParallelDownBlock2D: {in_channels} channels, {num_layers} layers, downsample={add_downsample}")

    def forward(self, hidden_states, index, temb=None, encoder_hidden_states=None,  **kwargs):
        output_states = ()
        content_feature = encoder_hidden_states[1][index]
        for resnet, local_attn, global_attn, gate in zip(self.resnets, self.local_attentions, self.global_attentions, self.content_gates):
            # 1. ResNet特征精炼
            hidden_states = resnet(hidden_states, temb)
            
            # Step 2: 双路注意力
            # 2a.  局部窗口注意力 (保护细节)
            h_local = local_attn(hidden_states)
            
            # 2b. 全局自注意力 (保持连贯)
            h_global = global_attn(hidden_states)

            # Step 3: 内容引导融合
            if gate is not None and content_feature is not None: 
                # 生成融合权重
                alpha = gate(hidden_states, content_feature)
                
                # 自适应融合: 
                # - 在笔画区域: 优先使用局部注意力 (保护细节)
                # - 在背景区域: 使用全局注意力 (保持流畅)
                hidden_states = alpha * h_local + (1 - alpha) * h_global
            else:
                # 无内容引导时简单平均
                hidden_states = 0.5 * h_local + 0.5 * h_global

            output_states += (hidden_states,)

        # 下采样
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states

# 重新设计的ParallelUpBlock2D - 类似StyleRSIUpBlock2D的结构
class ParallelUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels: int = 1,
        attention_type: str = "default",
        output_scale_factor: float = 1.0,
        add_upsample: bool = False,
        content_channel: int = 0, # ← 新增
        window_size: int = 4,  # ← 新增:  窗口注意力尺寸
    ):
        super().__init__()
        
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        self.num_layers = num_layers
        self.parallel = True

        resnets = []
        local_attentions = []   # ← 新增: 窗口注意力
        global_attentions = []  # ← 保留:  全局注意力
        content_gates = []      # ← 新增: 内容引导门控

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # ResNet块
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            # ← 新增: 局部窗口注意力 (保护细小笔画)
            local_attentions.append(
                WindowAttentionBlock(
                    channels=in_channels,
                    window_size=window_size,
                    num_heads=4,
                    num_groups=resnet_groups
                )
            )
            
            # 自注意力块 - 修正参数
            global_attentions. append(
                SelfAttentionBlock(
                    in_channels=out_channels,  # 修正：使用out_channels
                    n_heads=attn_num_head_channels,
                    d_head=out_channels // attn_num_head_channels,  # 修正：计算d_head
                    dropout=dropout,
                    num_groups=resnet_groups,
                )
            )

            # ← 新增: 内容引导门控
            if content_channel > 0:
                content_gates.append(
                    ContentGuidedGate(
                        in_channels=in_channels,
                        content_channels=content_channel,
                        num_groups=resnet_groups
                    )
                )
            else:
                content_gates.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.local_attentions = nn.ModuleList(local_attentions)
        self.global_attentions = nn.ModuleList(global_attentions)
        self.content_gates = nn.ModuleList(content_gates)


        #上采样层
        if add_upsample:
            print("Warning: ParallelUpBlock2D with add_upsample=True may break parallel processing")
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        print(f"Created ParallelUpBlock2D: {prev_output_channel}+{in_channels}->{out_channels}, {num_layers} layers, upsample={add_upsample}")

    def forward(self, hidden_states, index,res_hidden_states_tuple, temb=None, upsample_size=None, encoder_hidden_states=None,  **kwargs):
        for resnet, local_attn, global_attn, gate in zip(self.resnets, self.local_attentions, self.global_attentions, self.content_gates):
            # 处理跳跃连接
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            content_feature = encoder_hidden_states[3][index]

            # ✅ 处理跳跃连接尺寸不匹配（这是正常的！）
            if hidden_states. shape[2:] != res_hidden_states.shape[2:]:
                import torch.nn.functional as F
                res_hidden_states = F.interpolate(
                    res_hidden_states,
                    size=hidden_states.shape[2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 拼接跳跃连接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                # ResNet处理
                hidden_states = torch.utils.checkpoint. checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                # 局部窗口注意力处理
                h_local = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(local_attn), hidden_states
                )
                # 自注意力处理
                h_global = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(global_attn), hidden_states
                )

                #  内容引导融合
                if gate is not None and content_feature is not None: 
                    # 生成融合权重
                    alpha =  torch.utils.checkpoint.checkpoint(
                    create_custom_forward(gate), hidden_states, content_feature)
                    
                    # 自适应融合: 
                    # - 在笔画区域: 优先使用局部注意力 (保护细节)
                    # - 在背景区域: 使用全局注意力 (保持流畅)
                    hidden_states = alpha * h_local + (1 - alpha) * h_global
                else:
                    # 无内容引导时简单平均
                    hidden_states = 0.5 * h_local + 0.5 * h_global
            else:
                # 1. ResNet处理跳跃连接和特征精炼
                hidden_states = resnet(hidden_states, temb)
                
                # 局部窗口注意力处理
                h_local = local_attn(hidden_states)
                # 自注意力处理
                h_global = global_attn(hidden_states)
                if gate is not None and content_feature is not None: 
                    alpha = gate(hidden_states, content_feature)
                    hidden_states = alpha * h_local + (1 - alpha) * h_global
                else:
                    hidden_states = 0.5 * h_local + 0.5 * h_global

        # 上采样
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states  # 注意：平行块不返回offset，与普通UpBlock2D一致