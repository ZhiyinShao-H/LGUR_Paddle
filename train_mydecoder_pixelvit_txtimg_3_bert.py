from option.options import options, config
from data.dataloader import get_dataloader
# import torch
import random
from model.model import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import RankingLoss
# from torch import optim
import logging
import os
from test_during_train import test , test_part
# from torch.autograd import Variable
from model.DETR_model import  TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_bert
# import torch.nn as nn

import paddle
from paddle import optimizer
import paddle.nn as nn



logger = logging.getLogger()
logger.setLevel(logging.INFO)


# def save_checkpoint(state, opt):
#
#     filename = os.path.join(opt.save_path, 'model/best.pth.tar')
#     torch.save(state, filename)

def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pdparams')
    paddle.save(state, filename)

def load_checkpoint(opt):
    filename = os.path.join(opt.save_path, 'model/best.pdparams')
    state = paddle.load(filename)

    return state


def calculate_similarity(image_embedding, text_embedding):
    image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    similarity = paddle.matmul(image_embedding_norm, text_embedding_norm.t())

    return similarity

def calculate_similarity_part(numpart,image_embedding, text_embedding):
    image_embedding = paddle.concat([image_embedding[i] for i in range(numpart)],axis=1)
    text_embedding = paddle.concat([text_embedding[i] for i in range(numpart)], axis=1)
    image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    similarity = paddle.matmul(image_embedding_norm, text_embedding_norm.t())

    return similarity

def calculate_part_id(id_loss_fun,num_query,image_embedding,text_embedding):
    id_loss_ = []
    pred_i2t_ = []
    pred_t2i_ = []
    for i in range(num_query):
        id_loss, pred_i2t_local, pred_t2i_local = id_loss_fun[i](image_embedding[i], text_embedding[i], label)
        id_loss_.append(id_loss)
        pred_i2t_.append(pred_i2t_local)
        pred_t2i_.append(pred_t2i_local)
    id_loss_ = paddle.stack(id_loss_)
    id_loss = paddle.mean(id_loss_)
    pred_i2t_ = paddle.stack(pred_i2t_)
    pred_i2t_local = paddle.mean(pred_i2t_)
    pred_t2i_ = paddle.stack(pred_t2i_)
    pred_t2i_local = paddle.mean(pred_t2i_)

    return id_loss , pred_i2t_local, pred_t2i_local

if __name__ == '__main__':
    opt = options().opt
    opt.GPU_id = '1'
    paddle.device.set_device('gpu:{}'.format(opt.GPU_id))
    opt.data_augment = False
    opt.lr = 0.001
    opt.margin = 0.3

    opt.feature_length = 512

    opt.dataset = 'CUHK-PEDES'

    if opt.dataset == 'MSMT-PEDES':
        opt.pkl_root = '/home/zhiyin/tran_ACMMM/processed_data_singledata_ICFG/'
        opt.class_num = 3102
        opt.vocab_size = 2500
        opt.dataroot = '/home/zhiyin/ICFG_PEDES/ICFG_PEDES'
        # opt.class_num = 2802
        # opt.vocab_size = 2300
    elif opt.dataset == 'CUHK-PEDES':
        opt.pkl_root = '/home/zhiyin/tran_ACMMM/processed_data_singledata_CUHK/'  # same_id_new_
        opt.class_num = 11000
        opt.vocab_size = 5000
        opt.dataroot = '/home/zhiyin/CUHK-PEDES'

    opt.d_model = 1024
    opt.nhead = 4
    opt.dim_feedforward = 2048
    opt.normalize_before = False
    opt.num_encoder_layers = 3
    opt.num_decoder_layers = 3
    opt.num_query = 6
    opt.detr_lr = 0.0001
    opt.txt_detr_lr = 0.0001
    opt.txt_lstm_lr = 0.001
    opt.res_y = False
    opt.noself = False
    opt.post_norm = False
    opt.n_heads = 4
    opt.n_layers = 2
    opt.share_query = True
    opt.ViT_layer = 8
    opt.wordtype = 'bert'
    model_name = 'model_get_2'
    # model_name = 'test'
    opt.save_path = '../checkpoints/{}/'.format(opt.dataset) + model_name

    opt.epoch = 60
    opt.epoch_decay = [20, 40, 50]

    opt.batch_size = 64
    opt.start_epoch = 0
    opt.trained = False

    config(opt)
    opt.epoch_decay = [i - opt.start_epoch for i in opt.epoch_decay]

    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'
    id_loss_fun = nn.LayerList()
    for _ in range(opt.num_query):
        id_loss_fun.append(Id_Loss(opt).to(opt.device))
    ranking_loss_fun = RankingLoss(opt)
    network = TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_bert(opt).to(opt.device)



    logging.info("Model_size: {:.5f}M".format(sum(p.numel() for p in network.parameters()) / 1000000.0))
    ignored_params = (list(map(id, network.ImageExtract.parameters()))
                        + list(map(id, network.TextExtract.parameters()))
                        + list(map(id, network.conv_1X1_2.parameters()))
                      # + list(map(id, network.conv_1X1.parameters()))
                      # + list(map(id, network.TXTEncoder.parameters()))
                      # + list(map(id, network.TXTDecoder.parameters()))
                    )
    DETR_params = filter(lambda p: id(p) not in ignored_params, network.parameters())
    DETR_params = list(DETR_params)
    param_groups = [{'parameters': DETR_params, 'learning_rate': opt.detr_lr,"beta1":opt.adam_alpha,"beta2":opt.adam_beta},
                    # {'params': network.TXTEncoder.parameters(), 'lr': opt.txt_detr_lr},
                    # {'params': network.TXTDecoder.parameters(), 'lr': opt.txt_detr_lr},
                    {'parameters': network.ImageExtract.parameters(), 'learning_rate': opt.lr * 0.1,"beta1":opt.adam_alpha,"beta2":opt.adam_beta},
                    {'parameters': network.TextExtract.parameters(), 'learning_rate': opt.lr,"beta1":opt.adam_alpha,"beta2":opt.adam_beta},
                    {'parameters': network.conv_1X1_2.parameters(), 'learning_rate': opt.lr,"beta1":opt.adam_alpha,"beta2":opt.adam_beta},
                    # {'params': network.conv_1X1.parameters(), 'lr': opt.lr},
                    {'parameters': id_loss_fun.parameters(), 'learning_rate': opt.lr,"beta1":opt.adam_alpha,"beta2":opt.adam_beta}
                    ]


    optimize = optimizer.Adam(param_groups)

    test_best = 0
    test_history = 0
    if opt.trained:
        state = load_checkpoint(opt)
        network.load_state_dict(state['network'])
        test_best = state['test_best']
        test_history = test_best
        id_loss_fun.load_state_dict(state['W'])
        print('load the {} epoch param successfully'.format(state['epoch']))
    """
    network.eval()
    test_best = test(opt, 0, 0, network,
                     test_img_dataloader, test_txt_dataloader, test_best)
    network.train()
    exit(0)
    """

    scheduler = optimizer.lr.MultiStepDecay(optimize, opt.epoch_decay)

    for epoch in range(opt.start_epoch, opt.epoch):

        id_loss_sum = 0
        ranking_loss_sum = 0
        pred_i2t_local_sum = 0
        pred_t2i_local_sum = 0

        scheduler.step()

        for times, [image, label, caption_code, caption_length, caption_mask] in enumerate(train_dataloader):

            # network.eval()
            # test_best = test_part(opt, epoch + 1, times + 1, network,
            #                       test_img_dataloader, test_txt_dataloader, test_best)
            # network.train()
            image = image.to(opt.device)
            label = label.to(opt.device)
            caption_code = caption_code.to(opt.device).long()
            caption_mask = caption_mask.to(opt.device)


            image_embedding,image_embedding_dict, text_embedding ,text_embedding_dict= network(image, caption_code, caption_mask)

            id_loss , pred_i2t_local, pred_t2i_local = calculate_part_id(id_loss_fun,opt.num_query ,image_embedding, text_embedding)

            id_loss_dict, pred_i2t_local_dict, pred_t2i_local_dict = calculate_part_id(id_loss_fun,opt.num_query, image_embedding_dict, text_embedding_dict)

            similarity = calculate_similarity_part(opt.num_query,image_embedding, text_embedding)
            ranking_loss = ranking_loss_fun(similarity, label)
            similarity_dict = calculate_similarity_part(opt.num_query, image_embedding_dict, text_embedding_dict)
            ranking_loss_dict = ranking_loss_fun(similarity_dict, label)

            similarity_dict_text = calculate_similarity_part(opt.num_query, text_embedding, text_embedding_dict)
            ranking_loss_dict_text = ranking_loss_fun(similarity_dict_text, label)

            similarity_dict_image = calculate_similarity_part(opt.num_query, image_embedding, image_embedding_dict)
            ranking_loss_dict_image = ranking_loss_fun(similarity_dict_image, label)

            optimize.clear_grad()
            loss = (id_loss + ranking_loss + id_loss_dict + ranking_loss_dict + ranking_loss_dict_text + ranking_loss_dict_image)

            loss.backward()
            # network.eval()
            # test_best = test_part(opt, epoch + 1, times + 1, network,
            #                       test_img_dataloader, test_txt_dataloader, test_best)
            # network.train()
            optimize.step()
            # network.eval()
            # test_best = test_part(opt, epoch + 1, times + 1, network,
            #                       test_img_dataloader, test_txt_dataloader, test_best)
            # network.train()
            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d Setp: %d, ranking_loss: %.2f, id_loss: %.2f, ranking_loss_dict: %.2f, id_loss_dict: %.2f,ranking_loss_dict_text: %.2f, ranking_loss_dict_image: %.2f,"
                             "pred_i2t_local: %.3f pred_t2i_local %.3f"
                      % (epoch+1, opt.epoch, times+1, ranking_loss, id_loss, ranking_loss_dict,id_loss_dict,ranking_loss_dict_text,ranking_loss_dict_image,pred_i2t_local, pred_t2i_local))

            ranking_loss_sum += ranking_loss
            id_loss_sum += id_loss
            pred_i2t_local_sum += pred_i2t_local
            pred_t2i_local_sum += pred_t2i_local

        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)
        pred_i2t_local_avg = pred_i2t_local_sum / (times + 1)
        pred_t2i_local_avg = pred_t2i_local_sum / (times + 1)

        logging.info("Epoch: %d/%d , ranking_loss: %.2f, id_loss: %.2f,"
                     " pred_i2t_local: %.3f, pred_t2i_local %.3f "
                     % (epoch+1, opt.epoch, ranking_loss_avg, id_loss_avg, pred_i2t_local_avg, pred_t2i_local_avg))

        print(model_name)
        network.eval()
        test_best = test_part(opt, epoch + 1, times + 1, network,
                         test_img_dataloader, test_txt_dataloader, test_best)
        network.train()
        if test_best > test_history:
            state = {
                'test_best': test_best,
                'network': network.cpu().state_dict(),
                'optimizer': optimize.state_dict(),
                'W': id_loss_fun.cpu().state_dict(),
                'epoch': epoch + 1}

            save_checkpoint(state, opt)
            network.to(opt.device)
            id_loss_fun.to(opt.device)

            test_history = test_best

    logging.info('Training Done')





