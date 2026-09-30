def model_params(data_name, num_nodes, input_len, output_len):
    return {
        "num_nodes": num_nodes, "input_len": input_len, "output_len": output_len,
        "patch_len": 8, "stride": 4, "alpha": 0.3, "beta": 0.1,
        "ma_type": "ema", "use_revin": True,
    }
