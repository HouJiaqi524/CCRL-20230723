from train_model import train_xrmb
from utils_ import setup_seed

setup_seed(7)
# set the hyperparams for TD-CCRL in XRMB
# config = {
#     'a1': 1,
#     'a2': 1,
#     'a3': 1/1000,
# }


# set the hyperparams for CCRL(TD-CCRL) in MNIST
config = {
    'a1': 1,
    'a2': 1/3,
    'a3': 1/49,
}

train_xrmb(config)


