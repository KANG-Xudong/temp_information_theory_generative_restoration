import torch.nn.init as init


# initialize the parameters of the model with scalar value 0
def zeros_init(m):
    class_name = m.__class__.__name__
    if (class_name.find("Conv") != -1) or (class_name.find('Linear') != -1):
        init.zeros_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm") != -1:
        init.normal_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias.data, 0.0)


# initialize the parameters of the model with scalar value 1
def ones_init(m):
    class_name = m.__class__.__name__
    if (class_name.find("Conv") != -1) or (class_name.find('Linear') != -1):
        init.ones_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm") != -1:
        init.normal_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias.data, 0.0)


# initialize the parameters of the model using a normal distribution
# with a mean of 0 and a standard deviation of 0.02
def normal_init(m):
    class_name = m.__class__.__name__
    if (class_name.find("Conv") != -1) or (class_name.find('Linear') != -1):
        init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm") != -1:
        init.normal_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias.data, 0.0)


# initialize the parameters of the model using the xavier initialization with a normal distribution
def xavier_normal_init(m):
    class_name = m.__class__.__name__
    if (class_name.find("Conv") != -1) or (class_name.find('Linear') != -1):
        init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm") != -1:
        init.normal_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias.data, 0.0)


# initialize the parameters of the model using the kaiming initialization with a normal distribution
def kaiming_normal_init(m):
    class_name = m.__class__.__name__
    if (class_name.find("Conv") != -1) or (class_name.find('Linear') != -1):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm") != -1:
        init.normal_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias.data, 0.0)