import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding
from .layers.Causal_Conv import CausalConv
from .layers.Multi_Correlation import AutoCorrelation, AutoCorrelationLayer, CrossCorrelation, CrossCorrelationLayer, \
    MultiCorrelation
from .layers.Corrformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, \
    my_Layernorm, series_decomp
from easydict import EasyDict
from basicts.utils import data_transformation_4_xformer




class Corrformer(nn.Module):
    def __init__(self, **kwargs):
        super(Corrformer, self).__init__()
        configs = EasyDict(kwargs)
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.node_num = configs.node_num
        self.node_list = configs.node_list  # node_num = node_list[0]*node_list[1]*node_list[2]...
        self.output_attention = configs.output_attention
        self.target_features = configs.get("target_features")  # default the first feature as target

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Encoding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.root_path,
                                           configs.node_num, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.root_path,
                                           configs.node_num, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * configs.seq_len,
                                    num_channels=[configs.d_model // configs.n_heads * configs.seq_len] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
                                    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
                                    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))



    




    # def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
    
        
    #     torch.autograd.set_detect_anomaly(True) 
    #     def check_tensor(name, tensor, epoch=None, batch_seen=None):
    #         """检查 tensor 是否含有 NaN 或 Inf"""
    #         if tensor is None:
    #             return
    #         if torch.isnan(tensor).any():
    #             print(f"[NaN DETECTED] {name} (epoch={epoch}, batch={batch_seen}) shape={tuple(tensor.shape)}")
    #         if torch.isinf(tensor).any():
    #             print(f"[Inf DETECTED] {name} (epoch={epoch}, batch={batch_seen}) shape={tuple(tensor.shape)}")


    #     x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(
    #         history_data=history_data, 
    #         future_data=future_data, 
    #         start_token_len=self.label_len, 
    #         target_features=self.target_features
    #     )
    #     check_tensor("x_enc", x_enc, epoch, batch_seen)
    #     check_tensor("x_mark_enc", x_mark_enc, epoch, batch_seen)
    #     check_tensor("x_dec", x_dec, epoch, batch_seen)
    #     check_tensor("x_mark_dec", x_mark_dec, epoch, batch_seen)

    #     mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
    #     zeros = x_dec.new_zeros((x_dec.shape[0], self.pred_len, x_dec.shape[2]))
    #     seasonal_init, trend_init = self.decomp(x_enc)
    #     check_tensor("seasonal_init", seasonal_init, epoch, batch_seen)
    #     check_tensor("trend_init", trend_init, epoch, batch_seen)

    #     trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
    #     seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
    #     check_tensor("trend_init_after_cat", trend_init, epoch, batch_seen)
    #     check_tensor("seasonal_init_after_cat", seasonal_init, epoch, batch_seen)

    #     # Encoder reshape
    #     B, L, D = x_enc.shape
    #     _, _, C = x_mark_enc.shape
    #     x_enc = x_enc.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous().view(B * self.node_num, L, D // self.node_num)
    #     x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
    #     check_tensor("x_enc_after_reshape", x_enc, epoch, batch_seen)
    #     check_tensor("x_mark_enc_after_reshape", x_mark_enc, epoch, batch_seen)

    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)
    #     check_tensor("enc_out_after_embedding", enc_out, epoch, batch_seen)
    #     enc_out = self.encoder(enc_out, attn_mask=None)
    #     check_tensor("enc_out_after_encoder", enc_out, epoch, batch_seen)

    #     # Decoder reshape
    #     B, L, D = seasonal_init.shape
    #     _, _, C = x_mark_dec.shape
    #     seasonal_init = seasonal_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous().view(B * self.node_num, L, D // self.node_num)
    #     trend_init = trend_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous().view(B * self.node_num, L, D // self.node_num)
    #     x_mark_dec = x_mark_dec.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
    #     check_tensor("seasonal_init_after_reshape", seasonal_init, epoch, batch_seen)
    #     check_tensor("trend_init_after_reshape", trend_init, epoch, batch_seen)
    #     check_tensor("x_mark_dec_after_reshape", x_mark_dec, epoch, batch_seen)

    #     dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
    #     check_tensor("dec_out_after_embedding", dec_out, epoch, batch_seen)

    #     seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
    #     check_tensor("seasonal_part", seasonal_part, epoch, batch_seen)
    #     check_tensor("trend_part", trend_part, epoch, batch_seen)

    #     dec_out = trend_part + seasonal_part
    #     check_tensor("dec_out_after_add", dec_out, epoch, batch_seen)

    #     dec_out = dec_out[:, -self.pred_len:, :].view(B, self.node_num, self.pred_len, D // self.node_num).permute(0, 2, 1, 3).contiguous().view(B, self.pred_len, D)
    #     check_tensor("dec_out_final_reshape", dec_out, epoch, batch_seen)

    #     final = dec_out.unsqueeze(-1)
    #     check_tensor("final", final, epoch, batch_seen)

    #     return final


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        


        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data, future_data=future_data, start_token_len=self.label_len, target_features=self.target_features)


        # init & normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc * self.affine_weight.repeat(1, 1, self.node_num) + self.affine_bias.repeat(1, 1, self.node_num) # encond_in * node_num = 3840 or num of station
        # decomp
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = x_dec.new_zeros((x_dec.shape[0], self.pred_len, x_dec.shape[2]))
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input init
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        B, L, D = x_enc.shape # D = node_num * enc_in
        _, _, C = x_mark_enc.shape  # time_category
        x_enc = x_enc.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num) # \ means to next row, the last shape of x_enc is (node_num,time_step,enc_in)
        x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=None)
        # dec
        B, L, D = seasonal_init.shape
        _, _, C = x_mark_dec.shape
        seasonal_init = seasonal_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        trend_init = trend_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_dec = x_mark_dec.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len:, :] \
            .view(B, self.node_num, self.pred_len, D // self.node_num).permute(0, 2, 1, 3).contiguous() \
            .view(B, self.pred_len, D)  # B L D

        # scale back
        dec_out = dec_out - self.affine_bias.repeat(1, 1, self.node_num)
        dec_out = dec_out / (self.affine_weight.repeat(1, 1, self.node_num) + 1e-10)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        final = dec_out.view(dec_out.shape[0], dec_out.shape[1], self.node_num, -1) # B L N C
        final = final.view(final.shape[0], final.shape[1], -1)
        return final.unsqueeze(-1)
