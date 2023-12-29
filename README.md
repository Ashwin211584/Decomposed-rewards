This research project explores using ProtoPNet, a modified Convolutional Neural Network (CNN)
designed for classification, in the realm of Reinforcement Learning (RL) to enhance interpretability.
In safety-critical applications like autonomous driving, understanding the decision-making of
Neural Network controllers is vital. Ensuring the interpretability of the decision-making processes
of complex Neural Network controllers is crucial for effective collaboration, safety assurance, and
legal scrutiny. We integrate ProtoPNet with Decomposed Rewards, a framework breaking
down reward functions into reward components, enabling a better understanding of an RL agent’s
decisions. The modified ProtoPNet learns prototypical patterns for maximizing different reward
components, making them interpretable. The agent’s policy will be trained to maximize the total
reward in the environment, akin to standard RL tasks. The project is evaluated in the Car Racing
Box2D environment to assess the effectiveness and interpretability of this novel RL algorithm.
# MLproject - Prototypical Parts Network for Reinforcement Learning

## Instalation

1. Clone the project repository to your local machine.
```~/project_dir/ $ git clone https://github.com/ramprasad555/MLproject.git```

2. Activate virtual environment for the project (conda, venv, or other). Ensure python version 3.10+.

3. Install requirements from file.
```~/project_dir/ $ cd MLProject```
```~/project_dir/MLProject/ $ pip install -r requirements.txt```

4. Install Prototypical Parts Network code as git submodule.
```~/project_dir/MLProject/ $ git submodule add https://github.com/cfchen-duke/ProtoPNet```
```~/project_dir/MLProject/ $ git submodule init```

5. Modify Prototypical Parts Network submodule code to work with our repo.

    a. Fix relative import bug in `~/project_dir/MLProject/ProtoPNet/model.py`. Out of the box, the relative imports in this file (lines 6 to 11) resulted in an import error. If this occurs for you, simply add a `.` at the beginning of the imports to use proper relative import syntax:

    ```
    from .resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features

    from .densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features

    from .vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                            vgg19_features, vgg19_bn_features

    from .receptive_field import compute_proto_layer_rf_info_v2
    ```
    
    b. Fix the number of feature channels used in the CNN feature extractor (initial layers) of Prototypical Parts Network. This number was hardcoded in the original ProtoPNet repo, so there is no flexible option for changing it. We append the action values to the input of our critic network, and thus need 3 additional channels (for a total of 6 channels). The original Prototypical Parts Network only requires 3 channels, for the RGB values of the images the network is trained on.

    We will modify line 133 of the `~/project_dir/MLProject/ProtoPNet/resnet_features.py` file. We will change the value of the first argument to the `nn.Conv2d` method from `3` to `6`:

    ```
    self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    ```

6. Run the training script!
```~/project_dir/MLProject/ $ python train.py```
