# from torch import nn
from paddle import nn
from model.text_feature_extract import TextExtract, TextExtract_Bert_lstm
from torchvision import models
# import torch
# from torch.nn import init
from vit_pytorch import pixel_ViT, DECODER, PartQuery,mydecoder,mydecoder_DETR
from einops.layers.torch import Rearrange

import paddle
from model.model import ft_net_TransREID_local, ft_net_TransREID_local_smallDeiT, ft_net_TransREID_local_smallVit
# from VD_project import SOHO_Pre_VD
from einops import rearrange, repeat

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
#         # init.constant(m.bias.data, 0.0)
#     elif classname.find('BatchNorm1d') != -1:
#         init.normal(m.weight.data, 1.0, 0.02)
#         init.constant(m.bias.data, 0.0)
#     elif classname.find('BatchNorm2d') != -1:
#         init.constant(m.weight.data, 1)
#         init.constant(m.bias.data, 0)


# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.normal_(m.weight.data, std=0.001)

        # init.constant(m.bias.data, 0.0)

class conv(nn.Layer):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2D(input_dim, output_dim, kernel_size=1, bias_attr=False)]

        if BN:
            block += [nn.BatchNorm2D(output_dim)]
        if relu:
            block += [nn.ReLU()]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x



from paddle.vision.models import resnet50
class TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_bert(nn.Layer):

    def __init__(self, opt):
        super(TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_bert, self).__init__()

        self.opt = opt
        backbone = resnet50(pretrained=True)
        # backbone = ft_net_TransREID_local_smallVit()
        self.ImageExtract = backbone
        # self.TextExtract = TextExtract_nomax(opt)
        self.TextExtract = TextExtract_Bert_lstm(opt)
        self.avg_global = nn.AdaptiveMaxPool2D((1, 1))
        self.TXTDecoder = mydecoder(opt= opt,dim=384, depth=2, heads=6,
                               mlp_dim=512, pool='cls', patch_dim=384, dim_head=512,
                               dropout=0., emb_dropout=0.)
        self.TXTDecoder_2 = mydecoder(opt=opt, dim=384, depth=1, heads=6,
                                    mlp_dim=768, pool='cls', patch_dim=384, dim_head=64,
                                    dropout=0., emb_dropout=0.)
        self.pixel_to_patch = Rearrange('b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1=1, p2=1)
        self.patch_to_pixel = Rearrange('b (h w) c  -> b c h w', h=24, w=8)
        # self.conv_1X1_2 = conv(384, opt.feature_length)
        self.conv_1X1_2 = nn.LayerList()
        for _ in range(opt.num_query):
            self.conv_1X1_2.append(conv(384, opt.feature_length))
        self.pos_embed_image = paddle.create_parameter(paddle.randn(1, 48, opt.d_model))
        self.query_embed_image = paddle.create_parameter(paddle.randn(1, self.opt.num_query, 384))
        if opt.share_query == False:
            self.tgt_embed_image = paddle.create_parameter(paddle.randn(1, self.opt.num_query, 384))
        self.dict_feature = paddle.create_parameter(paddle.randn(1, 400, 384))
        self.mask = nn.Sequential(nn.Linear(384,1),
                                  nn.Sigmoid())

        # self.vd = SOHO_Pre_VD(8000, 384, decay=0.1, max_decay=0.99)
        # self.linear_768 = nn.Linear(2048, 384, bias=False)

    def forward(self, image, caption_id, text_mask):
        image_feature = self.ImageExtract(image)
        text_feature = self.TextExtract(caption_id, text_mask)
        image_feature_fusion = self.image_fusion(image_feature , text_feature , caption_id)
        image_feature_part, image_feature_part_dict = self.image_DETR(image_feature_fusion,image_feature)
        text_feature_part, text_feature_part_dict = self.text_DETR(text_feature, caption_id)
        return image_feature_part, image_feature_part_dict,text_feature_part,text_feature_part_dict


    def text_DETR(self,text_featuremap,caption_id):

        # text_featuremap = self.linear_768(text_featuremap)
        B, L, C = text_featuremap.shape
        dict_feature = self.dict_feature.repeat(B, 1, 1)
        tgt = text_featuremap
        ignore_kv_mask = (caption_id == 0)
        ignore_kv_mask = ignore_kv_mask[:, :text_featuremap.size(1)]
        ignore_kv_mask = paddle.logical_not(ignore_kv_mask)
        q_mask = paddle.zeros(B,self.opt.num_query).to(self.opt.device)
        q_mask = (q_mask == 0)
        # memory = self.TXTEncoder(tgt,mask = ignore_kv_mask)
        memory = tgt
        memory_dict = self.TXTDecoder_2(memory,dict_feature)
        if self.opt.share_query:
            tgt_embed_image = self.query_embed_image.repeat(B, 1, 1)
        else:
            tgt_embed_image = self.tgt_embed_image.repeat(B, 1, 1)

        hs = self.TXTDecoder(tgt_embed_image,memory,ignore_kv_mask,q_mask)
        hs_dict = self.TXTDecoder(tgt_embed_image, memory_dict, ignore_kv_mask, q_mask)

        text_part = []
        for i in range(self.opt.num_query):
            hs_i = self.conv_1X1_2[i](hs[:,i].unsqueeze(2).unsqueeze(2))
            text_part.append(hs_i.unsqueeze(0))
        text_part = paddle.concat(text_part, axis=0)

        text_part_dict = []
        for i in range(self.opt.num_query):
            hs_i = self.conv_1X1_2[i](hs_dict[:,i].unsqueeze(2).unsqueeze(2))
            text_part_dict.append(hs_i.unsqueeze(0))
        text_part_dict = paddle.concat(text_part_dict, axis=0)
        return text_part , text_part_dict

    def image_fusion(self,image_feature,text_feature, caption_id):
        B, P, C = image_feature.shape
        _, L, _ = text_feature.shape

        ignore_kv_mask = (caption_id == 0)
        ignore_kv_mask = ignore_kv_mask[:, :L]
        ignore_kv_mask = paddle.logical_not(ignore_kv_mask)
        q_mask = paddle.zeros(B, P).to(self.opt.device)
        q_mask = (q_mask == 0)
        mask = self.mask(image_feature)
        memory_mask = image_feature
        memory_dict = self.TXTDecoder_2(memory_mask, text_feature , ignore_kv_mask, q_mask)
        memory_dict = memory_dict * mask
        return memory_dict


    def image_DETR(self,image_featuremap_fusion , image_featuremap):
        B , P , C  = image_featuremap.shape
        dict_feature = self.dict_feature.repeat(B, 1, 1)

        memory = image_featuremap
        mask = self.mask(memory)
        memory_mask = memory
        memory_dict = self.TXTDecoder_2(memory_mask,dict_feature)
        memory_dict = memory_dict * mask
        query_embed_image = self.query_embed_image.repeat(B,1,1)
        if image_featuremap_fusion != None:
            hs = self.TXTDecoder(query_embed_image, image_featuremap_fusion)
        else:
            hs = self.TXTDecoder(query_embed_image, image_featuremap)

        hs_dict = self.TXTDecoder(query_embed_image, memory_dict)

        image_part = []
        for i in range(self.opt.num_query):
            hs_i = self.conv_1X1_2[i](hs[:,i].unsqueeze(2).unsqueeze(2))
            image_part.append(hs_i.unsqueeze(0))
        image_part = paddle.concat(image_part,axis=0)
        image_part_dict = []
        for i in range(self.opt.num_query):
            hs_i = self.conv_1X1_2[i](hs_dict[:,i].unsqueeze(2).unsqueeze(2))
            image_part_dict.append(hs_i.unsqueeze(0))
        image_part_dict = paddle.concat(image_part_dict, axis=0)
        return image_part , image_part_dict

    def img_embedding(self, image):
        image_feature = self.ImageExtract(image)
        image_feature_part , image_feature_part_dict= self.image_DETR(None, image_feature)
        return image_feature_part, image_feature_part_dict

    def txt_embedding(self, caption_id, text_mask):
        text_feature = self.TextExtract(caption_id, text_mask)
        text_feature_part, text_feature_part_dict= self.text_DETR(text_feature, caption_id )
        return text_feature_part, text_feature_part_dict