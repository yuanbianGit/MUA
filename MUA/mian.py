import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from utils.logger import setup_logger
from advers.GD import Generator, MS_Discriminator, Pat_Discriminator, GANLoss, weights_init,ResnetG
import torch
import torch.optim as optim
from advers.utils import save_checkpoint
import os.path as osp
from utils.metrics import R1_mAP_eval, R1_mAP
import numpy as np
import random
import torchvision
import torchvision.transforms as T
from DWT import *
from torchvision.utils import save_image
from torch.cuda import amp



# Imagenet_mean = [0.485, 0.456, 0.406]
# Imagenet_stddev = [0.229, 0.224, 0.225]


Imagenet_mean = [0.5, 0.5, 0.5]
Imagenet_stddev = [0.5, 0.5, 0.5]

vers_mean = [-1,-1,-1]
vers_std = [2,2,2]
DWT = DWT_2D_tiny(wavename='haar')
IDWT = IDWT_2D_tiny(wavename='haar')
pdist = torch.nn.PairwiseDistance(p=2)
lowFre_loss = torch.nn.SmoothL1Loss(reduction='sum')


def perturb_train(imgs, G, D, train_or_test='test',cam_ids=None,randE=None):
  # print('----------'+imgs.type())
  delta= G(imgs)
  # logger.info(delta.size())
  # NOTE 这里的L_norm能保障灰度图像加上还是灰度图像
  delta = L_norm(delta,train_or_test)
  # logger.info(delta.size()) torch.Size([128, 1, 288, 144])
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())
  # 加入噪声和没加入噪声的!
  _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))

  delta = delta * mask
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())
  for c in range(3):
    new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp((0.0 - Imagenet_mean[c]) / Imagenet_stddev[c],
                                                          (1.0 - Imagenet_mean[c]) / Imagenet_stddev[c]) # do clamping per channel

  if train_or_test == 'train':
    return new_imgs, mask
  elif train_or_test == 'test':
    return new_imgs, delta, mask

def L_norm(delta, mode='train'):
  '''
  大概意思就是clamp使得噪声符合约束！
  '''
  # 当时的transform ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
  delta = torch.clamp(delta,-8.0/255,8.0/255)
  if delta.size(1)==1:
      delta = delta.repeat(1,3,1,1)
  for c in range(delta.size(1)):
    delta.data[:,c,:,:] = (delta.data[:,c,:,:]) / Imagenet_stddev[c]
  return delta

def min_max_div(input):
    '''
    input (B,H,W)
    '''
    out_put=torch.zeros_like(input)
    for i in range(len(input)):
        min_vals= torch.min(input[i])
        max_vals= torch.max(input[i])

        # 最小-最大缩放，将x的范围缩放到[0, 1]
        scaled_x = (input[i] - min_vals) / (max_vals - min_vals)
        out_put[i] = scaled_x
    return out_put


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOPReID Testing")
    parser.add_argument(
        "--config_file", default="./configs/RGBNT201/ATT_TOP.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TOPReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()
    G_V = Generator(3, cfg.MODEL.G_CHANNEL, 32, norm='bn', beta=0.1).apply(weights_init).to(device)
    G_I = Generator(3, 1, 32, norm='bn', beta=0.1).apply(weights_init).to(device)
    G_T = Generator(3, cfg.MODEL.G_CHANNEL, 32, norm='bn', beta=0.1).apply(weights_init).to(device)
    D_V = MS_Discriminator(input_nc=6, norm='bn', temperature=-1.0, use_gumbel=False).apply(weights_init).to(device)
    D_I = MS_Discriminator(input_nc=6, norm='bn', temperature=-1.0, use_gumbel=False).apply(weights_init).to(device)
    D_T = MS_Discriminator(input_nc=6, norm='bn', temperature=-1.0, use_gumbel=False).apply(weights_init).to(device)

    optimizer_G_V = optim.Adam(G_V.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.5, 0.999))
    optimizer_D_V = optim.Adam(D_V.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.5, 0.999))

    optimizer_G_I = optim.Adam(G_I.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.5, 0.999))
    optimizer_D_I = optim.Adam(D_I.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.5, 0.999))

    optimizer_G_T = optim.Adam(G_T.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.5, 0.999))
    optimizer_D_T = optim.Adam(D_T.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.5, 0.999))

    criterionGAN = GANLoss()
    save_dir = cfg.OUTPUT_DIR
    best_hit, best_epoch = np.inf, 0
    scaler_G_V = amp.GradScaler()
    scaler_D_V = amp.GradScaler()
    scaler_G_I = amp.GradScaler()
    scaler_D_I = amp.GradScaler()
    scaler_G_T = amp.GradScaler()
    scaler_D_T = amp.GradScaler()

    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        model.eval()
        G_V.train()
        G_I.train()
        G_T.train()
        D_V.train()
        D_I.train()
        D_T.train()
        # for n_iter, (img, img_mask, vid, target_cam, target_view, _) in enumerate(train_loader):
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            # img_mask (B,1,h,w) 0 1
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            img_adv = {'RGB': 0,
                   'NI': 0,
                   'TI': 0}

            # img_mask = {'RGB': img_mask['RGB'].to(device),
            #        'NI': img_mask['NI'].to(device),
            #        'TI': img_mask['TI'].to(device)}

            with amp.autocast(enabled=True):

                img_adv['RGB'], mask_V = perturb_train(img['RGB'], G_V, D_V, train_or_test='train')
                img_adv['NI'], mask_I = perturb_train(img['NI'], G_I, D_I, train_or_test='train')
                img_adv['TI'], mask_T = perturb_train(img['TI'], G_T, D_T, train_or_test='train')

                pred_fake_pool_V, _ = D_V(torch.cat((img['RGB'], img_adv['RGB'].detach()), 1))
                pred_fake_pool_I, _ = D_I(torch.cat((img['NI'], img_adv['NI'].detach()), 1))
                pred_fake_pool_T, _ = D_T(torch.cat((img['TI'], img_adv['TI'].detach()), 1))

                loss_D_fake_V = criterionGAN(pred_fake_pool_V, False)
                loss_D_fake_I = criterionGAN(pred_fake_pool_I, False)
                loss_D_fake_T = criterionGAN(pred_fake_pool_T, False)

                num = cfg.SOLVER.IMS_PER_BATCH // 2

                pred_real_V, _ = D_V(torch.cat((img['RGB'][0:num, :, :, :], img['RGB'][num:, :, :, :].detach()), 1))
                pred_real_I, _ = D_I(torch.cat((img['NI'][0:num, :, :, :], img['NI'][num:, :, :, :].detach()), 1))
                pred_real_T, _ = D_T(torch.cat((img['TI'][0:num, :, :, :], img['TI'][num:, :, :, :].detach()), 1))

                loss_D_real_V = criterionGAN(pred_real_V, True)
                loss_D_real_I = criterionGAN(pred_real_I, True)
                loss_D_real_T = criterionGAN(pred_real_T, True)

                pred_fake_V, _ = D_V(torch.cat((img['RGB'], img_adv['RGB']), 1))
                pred_fake_I, _ = D_I(torch.cat((img['NI'], img_adv['NI']), 1))
                pred_fake_T, _ = D_T(torch.cat((img['TI'], img_adv['TI']), 1))

                loss_G_GAN_V = criterionGAN(pred_fake_V, True)
                loss_G_GAN_I = criterionGAN(pred_fake_I, True)
                loss_G_GAN_T = criterionGAN(pred_fake_T, True)

                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)

                output,g_V,g_I, g_T,f_V,f_I,f_T = model(img, label=target, cam_label=target_cam, view_label=target_view)

                g_Vc = g_V.reshape(-1, 4, 768).mean(1,keepdim=True).repeat(1, 4, 1).reshape(-1, 768)
                g_Tc = g_T.reshape(-1, 4, 768).mean(1,keepdim=True).repeat(1, 4, 1).reshape(-1, 768)
                g_Ic = g_I.reshape(-1, 4, 768).mean(1,keepdim=True).repeat(1, 4, 1).reshape(-1, 768)
                output_adv,g_V_adv,g_I_adv, g_T_adv,f_V_adv,f_I_adv,f_T_adv = model(img_adv, label=target, cam_label=target_cam, view_label=target_view)

                _, g_V_I, _ = model.NI(img['RGB'],layer=cfg.SOLVER.LAYER, cam_label=target_cam, view_label=target_view)
                _, g_V_I_adv, _ = model.NI(img_adv['RGB'],layer=cfg.SOLVER.LAYER, cam_label=target_cam, view_label=target_view)

                _, g_I_V, _ = model.RGB(img['NI'], layer=cfg.SOLVER.LAYER, cam_label=target_cam, view_label=target_view)
                _, g_I_V_adv, _ = model.RGB(img_adv['NI'], layer=cfg.SOLVER.LAYER, cam_label=target_cam, view_label=target_view)


                # TODO  进行跨模态的距离计算，如果说还有相似的跨模态特征，那么说明还是保留了一些特征的！
                # FIXME best performance
                # loss_G_ReID_V_feat = -cfg.SOLVER.FEAT_FACTOR * torch.mean(pdist(g_V_I.detach(), g_V_I_adv)) - cfg.SOLVER.FEAT_FACTOR_2 * (torch.mean(pdist(g_Ic.detach(), g_V_I_adv)) + torch.mean(pdist(g_Tc.detach(), g_V_I_adv))  + torch.mean(pdist(g_Vc.detach(), g_V_I_adv)))
                # loss_G_ReID_I_feat = -cfg.SOLVER.FEAT_FACTOR * torch.mean(pdist(g_I_V.detach(), g_I_V_adv)) - cfg.SOLVER.FEAT_FACTOR_2 * (torch.mean(pdist(g_Vc.detach(), g_I_V_adv)) + torch.mean(pdist(g_Tc.detach(), g_I_V_adv))  + torch.mean(pdist(g_Ic.detach(), g_I_V_adv)) )
                # loss_G_ReID_T_feat = 0 * torch.mean(pdist(g_Vc.detach(), g_T_adv))
                #
                loss_G_ReID_V_g = - cfg.SOLVER.ADV_FACTOR * (torch.mean(pdist(g_V.detach(), g_V_adv)))- cfg.SOLVER.ADV_FACTOR_2 *(torch.mean(pdist(g_Tc.detach(), g_V_adv))+torch.mean(pdist(g_Ic.detach(), g_V_adv))+torch.mean(pdist(g_Vc.detach(), g_V_adv)))
                loss_G_ReID_I_g = - cfg.SOLVER.ADV_FACTOR*0.5 * (torch.mean(pdist(g_I.detach(), g_I_adv)))- cfg.SOLVER.ADV_FACTOR_2 * (torch.mean(pdist(g_Tc.detach(), g_I_adv))+torch.mean(pdist(g_Vc.detach(), g_I_adv))+torch.mean(pdist(g_Ic.detach(), g_I_adv)))
                loss_G_ReID_T_g = - cfg.SOLVER.ADV_FACTOR * (torch.mean(pdist(g_T.detach(), g_T_adv)))- cfg.SOLVER.ADV_FACTOR_2* (torch.mean(pdist(g_Vc.detach(), g_T_adv))+torch.mean(pdist(g_Ic.detach(), g_T_adv))+torch.mean(pdist(g_Tc.detach(), g_T_adv)) )

                # loss_G_ReID_V_feat = -cfg.SOLVER.FEAT_FACTOR * torch.mean(pdist(g_V_I.detach(), g_V_I_adv)) - cfg.SOLVER.FEAT_FACTOR_2 * (torch.mean(pdist(g_Ic.detach(), g_V_I_adv)) + torch.mean(pdist(g_Tc.detach(), g_V_I_adv)) + torch.mean(pdist(g_Vc.detach(), g_V_I_adv)))
                # loss_G_ReID_I_feat = -cfg.SOLVER.FEAT_FACTOR * torch.mean(pdist(g_I_V.detach(), g_I_V_adv)) - cfg.SOLVER.FEAT_FACTOR_2 * (torch.mean(pdist(g_Vc.detach(), g_I_V_adv)) + torch.mean(pdist(g_Tc.detach(), g_I_V_adv)) + torch.mean(pdist(g_Ic.detach(), g_I_V_adv)))

                loss_G_ReID_V_feat = -cfg.SOLVER.FEAT_FACTOR * torch.mean(pdist(g_V_I.detach(), g_V_I_adv)) - cfg.SOLVER.FEAT_FACTOR_2 * (torch.mean(pdist(g_Vc.detach(), g_V_I_adv)) + torch.mean(pdist(g_Ic.detach(), g_V_I_adv))+torch.mean(pdist(g_Tc.detach(), g_V_I_adv)))
                loss_G_ReID_I_feat = -cfg.SOLVER.FEAT_FACTOR * torch.mean(pdist(g_I_V.detach(), g_I_V_adv)) - cfg.SOLVER.FEAT_FACTOR_2 * (torch.mean(pdist(g_Ic.detach(), g_I_V_adv)) + torch.mean(pdist(g_Vc.detach(), g_I_V_adv))+torch.mean(pdist(g_Tc.detach(), g_I_V_adv)))
                loss_G_ReID_T_feat = 0 * torch.mean(pdist(g_Vc.detach(), g_T_adv))

                # loss_G_ReID_V_g = - cfg.SOLVER.ADV_FACTOR * (torch.mean(pdist(g_V.detach(), g_V_adv))) - cfg.SOLVER.ADV_FACTOR_2 * (torch.mean(pdist(g_Tc.detach(), g_V_adv)) + torch.mean(pdist(g_Vc.detach(), g_V_adv)) + torch.mean(pdist(g_Ic.detach(), g_V_adv)))
                # loss_G_ReID_I_g = - cfg.SOLVER.ADV_FACTOR * 0.5 * (torch.mean(pdist(g_I.detach(), g_I_adv))) - cfg.SOLVER.ADV_FACTOR_2 * (torch.mean(pdist(g_Tc.detach(), g_I_adv)) + torch.mean(pdist(g_Vc.detach(), g_I_adv))+ torch.mean(pdist(g_Ic.detach(), g_I_adv)))
                # loss_G_ReID_T_g = - cfg.SOLVER.ADV_FACTOR * (torch.mean(pdist(g_T.detach(), g_T_adv))) - cfg.SOLVER.ADV_FACTOR_2 * (torch.mean(pdist(g_Vc.detach(), g_T_adv)) +torch.mean(pdist(g_Tc.detach(), g_T_adv)) + torch.mean(pdist(g_Ic.detach(), g_T_adv)))

                loss_G_ReID_V = loss_G_ReID_V_g + loss_G_ReID_V_feat
                loss_G_ReID_I = loss_G_ReID_I_g + loss_G_ReID_I_feat
                loss_G_ReID_T = loss_G_ReID_T_g + loss_G_ReID_T_feat

                loss_D_V = (loss_D_fake_V + loss_D_real_V)/2
                loss_D_I = (loss_D_fake_I + loss_D_real_I)/2
                loss_D_T = (loss_D_fake_T + loss_D_real_T)/2

                loss_G_V = loss_G_ReID_V+loss_G_GAN_V
                loss_G_I = loss_G_ReID_I+loss_G_GAN_I
                loss_G_T = loss_G_ReID_T+loss_G_GAN_T

            optimizer_G_V.zero_grad()
            scaler_G_V.scale(loss_G_V).backward()
            scaler_G_V.step(optimizer_G_V)
            scaler_G_V.update()

            optimizer_D_V.zero_grad()
            scaler_D_V.scale(loss_D_V).backward()
            scaler_D_V.step(optimizer_D_V)
            scaler_D_V.update()

            optimizer_G_I.zero_grad()
            scaler_G_I.scale(loss_G_I).backward()
            scaler_G_I.step(optimizer_G_I)
            scaler_G_I.update()

            optimizer_D_I.zero_grad()
            scaler_D_I.scale(loss_D_I).backward()
            scaler_D_I.step(optimizer_D_I)
            scaler_D_I.update()

            optimizer_G_T.zero_grad()
            scaler_G_T.scale(loss_G_T).backward()
            scaler_G_T.step(optimizer_G_T)
            scaler_G_T.update()

            optimizer_D_T.zero_grad()
            scaler_D_T.scale(loss_D_T).backward()
            scaler_D_T.step(optimizer_D_T)
            scaler_D_T.update()



            if (n_iter + 1) % 50 == 0:

                logger.info(
                    "===> Epoch[{}]({}/{}) loss_G_ReID_V: {:.4f} loss_G_ReID_I: {:.4f} loss_G_ReID_T: {:.4f}".format(
                        epoch, n_iter, len(train_loader), loss_G_ReID_V.item(), loss_G_ReID_I.item(),
                        loss_G_ReID_T.item()))
                logger.info(
                    "===> Epoch[{}]({}/{}) loss_G_ReID_V_g: {:.4f} loss_G_ReID_I_g: {:.4f} loss_G_ReID_T_g: {:.4f}".format(
                        epoch, n_iter, len(train_loader), loss_G_ReID_V_g.item(), loss_G_ReID_I_g.item(),
                        loss_G_ReID_T_g.item()))
                logger.info(
                    "===> Epoch[{}]({}/{}) loss_G_ReID_V_f: {:.4f} loss_G_ReID_I_f: {:.4f} loss_G_ReID_T_f: {:.4f}".format(
                        epoch, n_iter, len(train_loader), loss_G_ReID_V_feat.item(), loss_G_ReID_I_feat.item(),
                        loss_G_ReID_T_feat.item()))

                #
                # logger.info("===> Epoch[{}]({}/{}) optimizer_G_V: {:.4f} optimizer_G_I: {:.4f} optimizer_G_T: {:.4f}".format(
                #         epoch, n_iter, len(train_loader), optimizer_G_V.state_dict()['param_groups'][0]['lr'],  optimizer_G_I.state_dict()['param_groups'][0]['lr'],
                #     optimizer_G_T.state_dict()['param_groups'][0]['lr']))

        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator_adv = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator_adv.reset()
            evaluator.reset()
            G_V.eval()
            G_I.eval()
            G_T.eval()
            D_V.eval()
            D_I.eval()
            D_T.eval()

            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = {'RGB': img['RGB'].float().to(device),
                           'NI': img['NI'].float().to(device),
                           'TI': img['TI'].float().to(device)}

                    img_adv = {'RGB': 0,
                               'NI': 0,
                               'TI': 0}
                    with amp.autocast(enabled=True):
                        img_adv['RGB'], mask_V = perturb_train(img['RGB'], G_V, D_V, train_or_test='train')
                        img_adv['NI'], mask_I = perturb_train(img['NI'], G_I, D_I, train_or_test='train')
                        img_adv['TI'], mask_T = perturb_train(img['TI'], G_T, D_T, train_or_test='train')

                        camids = camids.to(device)
                        scenceids = target_view
                        target_view = target_view.to(device)
                        feat,_,_,_,_,_,_ = model(img, cam_label=camids, view_label=target_view)
                        feat_adv,_,_,_,_,_,_ = model(img_adv, cam_label=camids, view_label=target_view)

                        evaluator.update((feat, vid, camid))
                        evaluator_adv.update((feat_adv, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            cmc_adv, mAP_adv, _, _, _, _, _ = evaluator_adv.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP_adv))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_adv[r - 1]))

            is_best = mAP_adv <= best_hit
            if is_best:
                best_hit, best_epoch = mAP_adv, epoch
            logger.info("==> Best_epoch is {}, Best rank-1 {:.1%}".format(best_epoch, best_hit))
            save_checkpoint(G_V.state_dict(), is_best, 'G_V', osp.join(save_dir, 'G_V_ep' + str(epoch) + '.pth.tar'))
            save_checkpoint(D_V.state_dict(), is_best, 'D_V', osp.join(save_dir, 'D_V_ep' + str(epoch) + '.pth.tar'))

            save_checkpoint(G_I.state_dict(), is_best, 'G_I', osp.join(save_dir, 'G_I_ep' + str(epoch) + '.pth.tar'))
            save_checkpoint(D_I.state_dict(), is_best, 'D_I', osp.join(save_dir, 'D_I_ep' + str(epoch) + '.pth.tar'))

            save_checkpoint(G_T.state_dict(), is_best, 'G_T', osp.join(save_dir, 'G_T_ep' + str(epoch) + '.pth.tar'))
            save_checkpoint(D_T.state_dict(), is_best, 'D_T', osp.join(save_dir, 'D_T_ep' + str(epoch) + '.pth.tar'))