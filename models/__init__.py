from models import McMRSR

def create_model(opts):
    if opts.model_type == 'McMRSR':
        model = McMRSR.RecurrentModel(opts)
    else:
        raise NotImplementedError

    return model
