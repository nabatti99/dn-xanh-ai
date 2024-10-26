import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_small
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


class WasteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobnet = mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        train_nodes, eval_nodes = get_graph_node_names(self.mobnet)
        self.feature_extraction = create_feature_extractor(
            self.mobnet, return_nodes={"features.12": "mob_feature"}
        )
        self.conv1 = nn.Conv2d(576, 300, 3)
        # self.conv2 = nn.Conv2d(300,150, 3)
        self.fc1 = nn.Linear(10800, 30)
        # self.fc2 = nn.Linear(1000, 30)
        self.dr = nn.Dropout()

    def forward(self, x):
        feature_layer = self.feature_extraction(x)["mob_feature"]
        x = F.relu(self.conv1(feature_layer))
        # x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.dr(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        output = self.fc1(x)
        return output
