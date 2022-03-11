
def calculate_output(input_dim, conv):  # assume height x width padding, kernel_size, stride
    height = input_dim[1]
    width = input_dim[2]
    output_channels, kernel_size, stride, padding = conv
    output_height = int((height + 2 * padding - kernel_size[0]) / stride + 1)
    output_width = int((width + 2 * padding - kernel_size[1]) / stride + 1)
    return (output_channels, output_height, output_width)

config = {

    'conv1': [8, (7,7), 4, 0],  # output channels, kernel, stride, padding
    'conv2': [16, (5,5), 3, 0],  # output channels, kernel, stride, padding,
    'conv3': [32, (3,3), 2, 0],  # output channels, kernel, stride, padding
    'input_channels': 3, # replace with env param
    'image_size': (3, 180, 180) # cropped image
}

config['output_conv1'] = calculate_output(config['image_size'], config['conv1'])
config['output_conv2'] = calculate_output(config['output_conv1'], config['conv2'])
config['output_conv3'] = calculate_output(config['output_conv2'], config['conv3'])


