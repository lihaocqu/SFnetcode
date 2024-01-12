from model import SERSModel
from saliency import compute_saliency_maps

# Initialize the model
model = SERSModel()

# Assume your input data is stored in the variable `input_data`
# The shape of input data should match the requirement of your model
# For example, if your model expects the input shape to be (batch, seq, feature), 
# make sure `input_data`'s shape is like that.

saliency_map = compute_saliency_maps(model, input_data)

# Print out or visualize the saliency_map
print(saliency_map)