import functools
from typing import Dict
import torch
import torchvision.utils as vutils
from torch import nn

import torch.nn.functional as F
from capD.modules.embedding import RNN_ENCODER

#import lpips

def magp(img, sent, netD):
    img_inter = (img.data).requires_grad_()
    sent_inter= (sent.data).requires_grad_()
    output_dict = netD(img_inter)
    out = netD.logitor(output_dict["logit_features"], sent_inter)
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img_inter,sent_inter),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    gp = torch.mean((grad_l2norm) ** 6)
    return gp


class GANLoss():
    def __init__(self,cfg):
        self.type = cfg.TYPE
        self.d_loss_component = cfg.D_LOSS_COMPONENT.split(',')
        self.g_loss_component = cfg.G_LOSS_COMPONENT.split(',')
        self.logit_input = cfg.LOGIT_INPUT
        self.logit_stop_grad = cfg.LOGIT_STOP_GRAD
        self.cap_stop_grad = cfg.CAP_STOP_GRAD
        self.fa_feature = cfg.FA_FEATURE

        self.cap_coeff = cfg.CAP_COEFF 

        if "img_rec" in self.d_loss_component:
            #self.perceptual_fn = lpips.LPIPS(net="vgg").cuda()
            #self.perceptual_fn.net.requires_grad_ = False
            self.rec_fn = nn.MSELoss()
            

    def get_sent_embs(self, batch, text_encoder):
        if not isinstance(text_encoder, RNN_ENCODER):
            word_embs = text_encoder(batch["caption_tokens"])
            sent_embs = torch.sum(word_embs, dim=1)
            sent_embs = sent_embs / batch["caption_lengths"].unsqueeze(1)
            sent_embs = sent_embs * (sent_embs.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        else:
            tokens, tok_lens = batch["damsm_tokens"], batch["damsm_lengths"]
            hidden = text_encoder.init_hidden(tokens.size(0))
            _, sent_embs = text_encoder(tokens, tok_lens, hidden)
        return sent_embs

    def contra_loss(self, emb0, emb1):
        labels = torch.diag(torch.ones(emb0.size(0))).cuda().detach()
        emb0 = F.normalize(emb0, p=2, dim=1) 
        emb1 = F.normalize(emb1, p=2, dim=1)
        scores = torch.mm(emb0, emb1.transpose(0,1))
        s1 = F.log_softmax(scores, dim=1)
        s1 = s1 * labels
        s1 = - (s1.sum(1)).mean()
        s0 = F.log_softmax(scores, dim=0)
        s0 = s0 * labels 
        s0 = -(s0.sum(0)).mean()
        loss = s0 + s1
        return loss
        

    def compute_gp(self, batch, sent_embs, netD) -> Dict[str, torch.Tensor]:
        loss = {}
        errD_reg = magp(batch["image"], sent_embs, netD)
        loss.update(errD_reg = 2 * errD_reg)
        return loss

    def compute_d_loss(self, batch, sent_embs, fakes, netD) -> Dict[str, torch.Tensor]:
        # real
        loss = {}
        rec = None
        cap = None

        kwargs = {"image":batch["image"]} 
        if "cap" in self.d_loss_component:
            kwargs["batch"] = batch
            kwargs["cap_stop_grad"] = self.cap_stop_grad
        real_dict = netD(**kwargs)
        fake_dict = netD(fakes.detach())
        if "logit" in self.d_loss_component:
            real_features = real_dict["logit_features"]
            fake_features = fake_dict["logit_features"]
            if self.logit_stop_grad:
                real_features = real_features.detach()
                fake_features = fake_features.detach() 
            real_output = netD.logitor(real_features, sent_embs)
            mis_output = netD.logitor(real_features[:-1], sent_embs[1:])
            fake_output = netD.logitor(fake_features, sent_embs)
            if self.type == "hinge":
                errD_real =  F.relu(1.0 - real_output).mean()
                errD_mis = F.relu(1.0 + mis_output).mean()
                errD_fake = F.relu(1.0 + fake_output).mean()
            else:
                raise NotImplementedError

            loss.update(
                errD_real = errD_real,
                errD_mis = 0.5 * errD_mis,
                errD_fake = 0.5 * errD_fake,
            )

        if "cap" in self.d_loss_component:
            errD_cap = real_dict["cap_loss"]
            cap = real_dict["predictions"]
            loss.update(errD_cap = self.cap_coeff * errD_cap)
        
        if "img_rec" in self.d_loss_component: 
            rec = netD.img_decoder(real_dict["dec_features"])
            #errD_rec = self.perceptual_fn(rec, batch["image"].detach()).mean()
            errD_rec = self.rec_fn(rec, batch["image"].detach())
            loss.update(errD_rec = errD_rec)

        if "sent_contra" in self.d_loss_component:
            raise NotImplementedError
            img_feat = netD.logitor.get_contra_img_feat(real_dict[self.logit_input])
            sent_feat = netD.logitor.get_contra_sent_feat(sent_embs)
            errD_sent = self.contra_loss(img_feat, sent_feat)
            loss.update(errD_sent = errD_sent)
        
        

        return loss, rec, cap

    def compute_g_loss(self, batch, sent_embs, fakes, netD) -> Dict[str, torch.Tensor]:
        # real
        loss = {}
        cap = None

        # Todo: update emb
        fake_kwargs = {"image":fakes}
        if 'cap' in self.g_loss_component or self.fa_feature == "projected_visual_features":
            fake_kwargs.update(batch=batch)
        fake_dict = netD(**fake_kwargs)

        if 'logit' in self.g_loss_component:
            fake_output = netD.logitor(fake_dict["logit_features"], sent_embs)
            if self.type == "hinge":
                errG_fake = -fake_output.mean() 
            else:
                raise NotImplementedError
            loss.update(errG_fake = errG_fake)

        if 'cap' in self.g_loss_component:
            errG_cap = fake_dict["cap_loss"]
            cap = fake_dict["predictions"]
            loss.update(errG_cap = self.cap_coeff * errG_cap)

        if 'img_fa' in self.g_loss_component:
            with torch.no_grad():
                real_dict=netD(batch["image"])
                real_feat = real_dict[self.fa_feature]
            fake_feat = fake_dict[self.fa_feature]
            errG_fa = torch.abs(fake_feat-real_feat.detach()).mean()
            loss.update(errG_fa=errG_fa)

        if 'img_contra' in self.g_loss_component:
            raise NotImplementedError
            kwargs = {"image":batch["image"]}
            if self.fa_feature == "projected_visual_features":
                kwargs.update(batch=batch)
            with torch.no_grad():
                real_dict = netD(**kwargs)
                real_feat = F.adaptive_avg_pool2d(real_dict[self.logit_input], (1,1))
                real_feat = real_feat.view(real_feat.size(0), -1).detach()
            fake_feat = F.adaptive_avg_pool2d(fake_dict[self.logit_input], (1,1)).view(real_feat.size(0), -1)
            errG_img = self.contra_loss(fake_feat, real_feat) 
            loss.update(errG_img = 0.2 * errG_img)

        if "sent_contra" in self.g_loss_component:
            raise NotImplementedError
            with torch.no_grad():
                sent_feat = netD.logitor.get_contra_sent_feat(sent_embs)
                sent_feat = sent_feat.detach()
            fake_feat = netD.logitor.get_contra_img_feat(fake_dict[self.logit_input])
            errG_sent = self.contra_loss(fake_feat, sent_feat)
            loss.update(errG_sent = errG_sent)

        
        return loss, cap

    def accumulate_loss(self, loss_dict):
        loss = 0.
        for key in loss_dict:
            loss += loss_dict[key] 
        return loss

   







