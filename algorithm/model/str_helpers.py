

def get_size_str(batch_size,nb_inputs,nb_outputs):
            return f"{batch_size}_{nb_inputs}_{nb_outputs}"

def get_inst(nr,padding):
    return str(nr + 1).zfill(padding)