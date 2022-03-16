import torch

def calculate_output(input_dim, conv):  # assume height x width padding, kernel_size, stride
    height = input_dim[1]
    width = input_dim[2]
    output_channels, kernel_size, stride, padding = conv
    output_height = int((height + 2 * padding - kernel_size[0]) / stride + 1)
    output_width = int((width + 2 * padding - kernel_size[1]) / stride + 1)
    return (output_channels, output_height, output_width)

config = {
    # General settings
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Agent defaults
    'discount': 0.99,
    'tau':0.005,
    'policy_noise':0.2,
    'noise_clip':0.5,
    'policy_freq': 2,
    # network settings:
    'conv1': [32, (7,7), 4, 0],  # output channels, kernel, stride, padding
    'conv2': [32, (5,5), 3, 0],  # output channels, kernel, stride, padding,
    'conv3': [32, (3,3), 2, 0],  # output channels, kernel, stride, padding
    'input_channels': 3, # replace with env param
    'image_size': (3, 180, 180), # cropped image
    'batch_size': 64,
    'BVAE_hidden':256,
    'BVAE_latent':128,
    'agent_hidden':256,
    # Action settings
    'action_dim': 9,
    'max_action': [1,1,1,1,1,1,1,1.57,1.57],
    'min_action': [-1,-1,-1,-1,-1,-1,-1,0,0],
    'training_agent_freq':5,
    'training_BVAE_freq':10,
    'start_timesteps':10000,
    'BVAE_pretrain_steps':15,
}

config['output_conv1'] = calculate_output(config['image_size'], config['conv1'])
config['output_conv2'] = calculate_output(config['output_conv1'], config['conv2'])
config['output_conv3'] = calculate_output(config['output_conv2'], config['conv3'])

# Actions: OrderedDict([('cartesian_command', array([-0.1921648 , -0.20408944,  0.51184336,  0.44491961,  0.03696635,
#         0.39647419, -0.53149232])), ('gripper_command', array([0.72575952, 1.38392144])), ('render', array([1],
#         dtype=int8))])
# Action space: Dict(cartesian_command:Box(-1.0, 1.0, (7,), float64), gripper_command:Box(0.0, 1.5707963267948966,
# (2,), float64), render:MultiBinary(1))

