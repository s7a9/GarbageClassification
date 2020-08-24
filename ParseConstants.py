from  torchvision.models import resnet18, resnet50, resnet101
import CBAMModels

def get_model(model_name, use_cbam):
    if not use_cbam:
        if model_name == 'resnet18':
            return resnet18(pretrained=True, progress=True)
        elif model_name == 'resnet50':
            return resnet50(pretrained=True, progress=True)
        elif model_name == 'resnet101':
            return resnet101(pretrained=True, progress=True)
    else:
        if model_name == 'resnet18':
            return CBAMModels.resnet18(pretrained=True, progress=True)
        elif model_name == 'resnet50':
            return CBAMModels.resnet50(pretrained=True, progress=True)
        elif model_name == 'resnet101':
            return CBAMModels.resnet101(pretrained=True, progress=True)
    print("Unexpected model name!!")
    return None

from torch import optim
optim_dict = {
    'SGD':      optim.SGD,
    'Adadelta': optim.Adadelta,
    'Adam':     optim.Adam
}
def get_optimizer(model, lr, scheduler_name, optim_name):
    optimizer = None
    if optim_name == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr= lr, rho=0.9, eps=1e-06, weight_decay=0)
    elif optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr= lr)
    else: #optim_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= lr, momentum= 0.9)
    if scheduler_name == 'StepLR':
        return optimizer, optim.lr_scheduler.StepLR(optimizer, 30)
    elif scheduler_name == 'ExpLr':
        return optimizer, optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    elif scheduler_name == 'CosLR':
        return optimizer, optim.lr_scheduler.CosineAnnealingLR(optimizer, 8)
    elif scheduler_name == 'Plateau':
        return optimizer, optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    else: #scheduler_name == 'None':
        return optimizer