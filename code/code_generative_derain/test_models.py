import os
import yaml
import torch
import importlib

def model_inference_test(model, input_shape=(7, 3, 256, 256)):
    input = torch.ones(input_shape)
    output = model(input)
    if input.size() == output.size():
        print("Equal Size:", output.size())
    else:
        print("Inconsistant! Output size is NOT equal to the input size.", output.size())


def model_loading_test(network_name):
    network_file = os.path.join('src', 'models', network_name + '.py')
    assert os.path.exists(network_file), "File not found! {}".format(network_file)
    module = importlib.import_module('src.models.{}'.format(network_name))
    if hasattr(module, 'Generator'):
        return module.Generator()
    elif hasattr(module, 'Discriminator'):
        return module.Discriminator()
    else:
        raise AttributeError("None of the class named \"Generator\" or class \"Discriminator\" were found!")


def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def main():
    generator_networks = yaml.safe_load(open(os.path.join('config', 'networks.yaml'), 'r'))['generator']
    for network in generator_networks:
        print("\n{}:".format(network))
        model = model_loading_test(network)
        model_inference_test(model, input_shape=(2, 3, 256, 256))
        num_parameters= count_parameters(model)
        print("{} parameters".format(num_parameters))


if __name__ == '__main__':
    main()