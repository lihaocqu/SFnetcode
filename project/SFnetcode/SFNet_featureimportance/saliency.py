import torch
from torch.nn.functional import normalize

def compute_saliency_maps(model, input_data):
    # Set the model to evaluation mode
    model.eval()

    # Enable gradient compute for input_data
    input_data.requires_grad_()

    # Perform a forward pass and store the output
    output = model(input_data)

    # Compute gradient of the output with respect to the input
    output.backward(torch.ones_like(output))

    # Compute the input gradient
    grad = input_data.grad

    # Compute weight times gradient
    saliency_map = grad * input_data
    saliency_map = torch.abs(saliency_map)

    # Compute sum of contributions from each neuron
    neurons_contributions = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.LSTM)):
            grad = torch.autograd.grad(output, layer.weight, retain_graph=True)[0]
            contribution = grad * layer.weight.data
            contribution = torch.abs(contribution).sum()
            neurons_contributions.append(contribution)

    neurons_contributions_sum = torch.Tensor(neurons_contributions).sum()

    # Combine the saliency map with sum of neuron contributions for final feature importance
    feature_importance = saliency_map + neurons_contributions_sum

    # Normalize feature importance before returning
    feature_importance = normalize(feature_importance, dim=0)

    return feature_importance