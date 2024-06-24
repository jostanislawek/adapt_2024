import os
import timm
import fastai
from fastai.vision.all import *
from timm import create_model
from fastai.vision.learner import _update_first_layer
import torchvision
import torch.backends.cudnn as cudnn
import paths as p
import helper_functions as hf
import model_helpers as mh

def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=True, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

def create_timm_model(arch:str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model

def check_freeze():
    vision_learner.freeze()
    frozen = filter(lambda p: not p.requires_grad, vision_learner.model.parameters())
    frozen = sum([np.prod(p.size()) for p in unfrozen_params])
    model_parameters = filter(lambda p: p.requires_grad, vision_learner.model.parameters())
    unfrozen = sum([np.prod(p.size()) for p in model_parameters])

def train_model(dls, model, n_out, tune_no, lr=0.1, loss_func=None, opt_func=Adam, metrics="accuracy"):
    model = create_timm_model(model, n_out, default_split, pretrained=True)
    learn  = vision_learner(dls, model, normalize=True, n_out=5, loss_func=loss_func, opt_func=opt_func, lr=0.1)
    model.fine_tune(tune_no, cbs=ShowGraphCallback())
    return learn
    # print("SHOW TRAINING LOOP")
    # model.show_training_loop()

    # Get validation metrics
    if metrics is not None:
        if hasattr(dls, 'valid'):
            val_results = model.show_results(ds_idx=1, dl=dls.valid)
            print("Validation Metrics:")
            if val_results is not None:
                for metric in metrics:
                    if metric in val_results:
                        print(f"{metric}: {val_results[metric]}")
                    else:
                        print(f"Metric '{metric}' not found in validation results.")
            else:
                print("No validation results.")
        else:
            print("Validation dataset not found. Metrics cannot be calculated.")
    else:
        print("No metrics specified.")

    return model

def get_model_params(trained_model):
    # Accessing model parameters
    for name, param in trained_model.model.named_parameters():
        print(f"Parameter: {name}, Value: {param}")

def evaluate_model(model):
    """ Prints the model architecture """
    print(model.eval())
    return model

def save_model(model, path):
    model_path = os.path.join(path, "model.pt")
    torch.save(model, model_path)