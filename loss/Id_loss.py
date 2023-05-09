# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 17(rd), 2019 at 15:33

@author: zifyloo
"""

import paddle
import paddle.nn as nn
# from torch.nn.parameter import Parameter
# from paddle.nn import init


# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.normal_(m.weight.data, std=0.001)
#         init.constant_(m.bias.data, 0.0)


class classifier(nn.Layer):

    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)
        # self.block.apply(weights_init_classifier)

    def forward(self, x):
        x = self.block(x)
        return x


class Id_Loss(nn.Layer):

    def __init__(self, opt):
        super(Id_Loss, self).__init__()

        self.opt = opt

        self.W = classifier(opt.feature_length, opt.class_num)
        # self.W_txt = classifier(opt.feature_length, opt.class_num)

    def calculate_IdLoss(self, image_embedding, text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t = self.W(image_embedding)
        score_t2i = self.W(text_embedding)
        Lipt_local = criterion(score_i2t, label)
        Ltpi_local = criterion(score_t2i, label)
        pred_i2t = paddle.mean((paddle.argmax(score_i2t, axis=1) == label))
        pred_t2i = paddle.mean((paddle.argmax(score_t2i, axis=1) == label))
        loss = (Lipt_local + Ltpi_local)

        return loss, pred_i2t, pred_t2i

    def forward(self, image_embedding, text_embedding, label):

        loss, pred_i2t, pred_t2i = self.calculate_IdLoss(image_embedding, text_embedding, label)

        return loss, pred_i2t, pred_t2i


class Id_Loss_2(nn.Layer):

    def __init__(self, opt):
        super(Id_Loss_2, self).__init__()

        self.opt = opt

        self.W = classifier(opt.feature_length, opt.class_num)
        # self.W_txt = classifier(opt.feature_length, opt.class_num)

    def calculate_IdLoss(self, image_embedding, text_embedding, label):

        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t = self.W(image_embedding)
        score_t2i = self.W(text_embedding)
        Lipt_local = criterion(score_i2t, label)
        Ltpi_local = criterion(score_t2i, label)
        pred_i2t = paddle.mean((paddle.argmax(score_i2t, axis=1) == label))
        pred_t2i = paddle.mean((paddle.argmax(score_t2i, axis=1) == label))
        # loss = (Lipt_local + Ltpi_local)

        return Lipt_local, Ltpi_local,pred_i2t, pred_t2i

    def forward(self, image_embedding, text_embedding, label):

        Lipt_local, Ltpi_local, pred_i2t, pred_t2i = self.calculate_IdLoss(image_embedding, text_embedding, label)

        return Lipt_local, Ltpi_local, pred_i2t, pred_t2i

