import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

C = 3
# in_channels = 19*(2*C + 1)
in_channels = 60

class AlexNet(nn.Module):

    def __init__(self, num_classes=8):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(128,momentum=0.1).train(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128,momentum=0.1).train(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256,momentum=0.1).train(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256,momentum=0.1).train(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(512,momentum=0.1).train(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512,momentum=0.1).train(),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=1, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def alexnet(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

def xavier_init(self):
	for m in self.modules():
		if isinstance(m, nn.Conv2d):
			init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
			m.bias.data.fill_(0.01)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.fill_(0.01)

