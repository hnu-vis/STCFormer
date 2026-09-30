def model_params(data_name, num_nodes, input_len, output_len):
    return {
        "num_nodes": num_nodes, "input_len": input_len, "output_len": output_len,
        "cycle_len": 24, "d_model": 128, "dropout": 0.1, "num_heads": 4,
        "attention_group_size": 256, "use_revin": True,
    }
