import os


def model_params(data_name, num_nodes, input_len, output_len):
    return {
        "num_nodes": num_nodes, "input_len": input_len, "output_len": output_len,
        "root_path": os.path.join("datasets", data_name),
        "d_model": 64, "num_layers": 2, "dropout": 0.1,
        "use_absolute_position": True,
    }
