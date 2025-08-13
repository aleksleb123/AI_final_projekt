import torch


# Forces torch to initialize cuDNN
# From StackOverflow https://stackoverflow.com/questions/66588715
def force_init_cudnn(dev=torch.device("cuda:0")):
    s = 32
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )
