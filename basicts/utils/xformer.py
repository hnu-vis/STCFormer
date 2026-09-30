import torch




def data_transformation_4_xformer(history_data: torch.Tensor, future_data: torch.Tensor, start_token_len: int, target_features: list):


    if len(target_features) == 1:
        # get the x_enc
        x_enc = history_data[..., 0]            # B, L1, N
        # get the corresponding x_mark_enc
        # following previous works, we re-scale the time features from [0, 1) to to [-0.5, 0.5).
        x_mark_enc = history_data[:, :, 0, 1:] - 0.5    # B, L1, C-1

        # get the x_dec
        if start_token_len == 0:
            x_dec = torch.zeros_like(future_data[..., 0])     # B, L2, N
            # get the corresponding x_mark_dec
            x_mark_dec = future_data[..., :, 0, 1:] - 0.5                 # B, L2, C-1
            return x_enc, x_mark_enc, x_dec, x_mark_dec
        else:
            x_dec_token = x_enc[:, -start_token_len:, :]            # B, start_token_length, N
            x_dec_zeros = torch.zeros_like(future_data[..., 0])     # B, L2, N
            x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)    # B, (start_token_length+L2), N
            # get the corresponding x_mark_dec
            x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]            # B, start_token_length, C-1
            x_mark_dec_future = future_data[..., :, 0, 1:] - 0.5          # B, L2, C-1
            x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)    # B, (start_token_length+L2), C-1

        return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()
    else:
        # get the x_enc
        x_enc = history_data[..., target_features].transpose(1,2).contiguous().flatten(start_dim=-2)  # B, N, L1*D
        # get the corresponding x_mark_enc
        # following previous works, we re-scale the time features from [0, 1) to to [-0.5, 0.5).
        x_mark_enc = history_data[:, :, 0, len(target_features):] - 0.5    # B, L1, C-1

        # get the x_dec
        if start_token_len == 0:
            x_dec = torch.zeros_like(future_data[..., target_features].transpose(1,2).contiguous().flatten(start_dim=-2))     # B, L2, N*D
            # get the corresponding x_mark_dec
            x_mark_dec = future_data[..., :, 0, len(target_features):] - 0.5                 # B, L2, C-1
            return x_enc, x_mark_enc, x_dec, x_mark_dec
        else:
            x_dec_token = x_enc[:, -start_token_len:, :]            # B, start_token_length, N*D
            x_dec_zeros = torch.zeros_like(future_data[..., target_features].transpose(1,2).contiguous().flatten(start_dim=-2))     # B, L2, N*D
            x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)    # B, (start_token_length+L2), N*D
            # get the corresponding x_mark_dec
            x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]            # B, start_token_length, C-1
            x_mark_dec_future = future_data[..., :, 0, len(target_features):] - 0.5          # B, L2, C-1
            x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)    # B, (start_token_length+L2), C-1

        return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()