""" Full assembly of the parts to form the complete network """
import torch
import numpy as np
from .p2p_parts import *
from .p2p_resnet import resnet50


class P2PNet(nn.Module):
    def __init__(self, num_classes, pyramid_level, row, col):
        super(P2PNet, self).__init__()
        self.backbone = resnet50(pretrained=True)

        self.anchor_points = AnchorPoints(pyramid_level, row, col)
        self.regression_branch = RegressionModel(num_features_in=2048, num_anchor_points=int(row * col),
                                                 feature_size=256)
        self.detection_branch = DetectionModel(num_features_in=2048, num_anchor_points=int(row * col),
                                               num_classes=2, feature_size=256)
        self.classification_branch = ClassificationModel(num_features_in=2048, num_anchor_points=int(row * col),
                                                         num_classes=num_classes, feature_size=256)

    def forward(self, x):
        x = self.backbone(x)
        batch_size = x.shape[0]

        anchor_points = self.anchor_points(x.shape[2:]).repeat(batch_size, 1, 1)
        deltas = self.regression_branch(x)
        fore_logits = self.detection_branch(x)
        pred_logits = self.classification_branch(x)
        coords = deltas + anchor_points
        return {'pred_points': coords, 'pred_logits': pred_logits, 'fore_logits': fore_logits}


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points, feature_size):
        super(RegressionModel, self).__init__()

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)

        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points, num_classes, feature_size):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)

        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes)


class DetectionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points, num_classes, feature_size):
        super(DetectionModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)

        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_classes)


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_level, row, line):
        super(AnchorPoints, self).__init__()

        self.stride = 2 ** pyramid_level
        self.row = row
        self.line = line

    def forward(self, image_shape):
        anchor_points = self.generate_anchor_points(self.stride, row=self.row, line=self.line)
        all_anchor_points = self.shift(image_shape, self.stride, anchor_points)
        return torch.from_numpy(all_anchor_points).float().cuda()

    @staticmethod
    def generate_anchor_points(stride, row, line):
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        return anchor_points

    @staticmethod
    def shift(shape, stride, anchor_points):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]
        K = shifts.shape[0]
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((1, K * A, 2))
        return all_anchor_points
