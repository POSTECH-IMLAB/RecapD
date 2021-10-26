import functools
from typing import Dict
import torch
from torch import nn

import torch.nn.functional as F

def magp(img, sent, netD):
    img_inter = (img.data).requires_grad_()
    sent_inter= (sent.data).requires_grad_()
    output_dict = netD(img_inter)
    out = netD.logitor(output_dict["visual_features"], sent_inter)
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
        self.fa_feature = cfg.FA_FEATURE

    def compute_d_loss(self, batch, text_encoder, netG, netD) -> Dict[str, torch.Tensor]:
        # real
        loss = {}
        with torch.no_grad():
            word_embeddings = text_encoder(batch["caption_tokens"])
            sent_embeddings = torch.sum(word_embeddings, dim=1)
            sent_embeddings = sent_embeddings / batch["caption_lengths"].unsqueeze(1)
            # normalize
            sent_embeddings = sent_embeddings * (sent_embeddings.square().mean(1, keepdim=True) + 1e-8).rsqrt()

        with torch.no_grad():
            fakes = netG(batch["z"], sent_embeddings) 

        real_dict = netD(
            image=batch["image"], 
            caption_tokens=batch["caption_tokens"], 
            caption_lengths=batch["caption_lengths"],
            noitpac_tokens=batch["noitpac_tokens"]
        )
        fake_dict = netD(fakes)
        if 'logit' in self.d_loss_component:
            real_output = netD.logitor(real_dict[self.logit_input], sent_embeddings)
            mis_output = netD.logitor(real_dict[self.logit_input][:-1], sent_embeddings[1:])
            fake_output = netD.logitor(fake_dict[self.logit_input], sent_embeddings)
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

        if 'magp' in self.d_loss_component:
            errD_reg = magp(batch["image"], sent_embeddings, netD)
            loss.update(errD_reg = 2 * errD_reg)
        
        if 'cap' in self.d_loss_component:
            errD_cap = real_dict["cap_loss"]
            loss.update(errD_cap = errD_cap)

        return loss

    def compute_g_loss(self, batch, text_encoder, netG, netD) -> Dict[str, torch.Tensor]:
        # real
        loss = {}
        word_embeddings = text_encoder(batch["caption_tokens"])
        sent_embeddings = torch.sum(word_embeddings, dim=1)
        sent_embeddings = sent_embeddings / batch["caption_lengths"].unsqueeze(1)
        # normalize
        sent_embeddings = sent_embeddings * (sent_embeddings.square().mean(1, keepdim=True) + 1e-8).rsqrt()

        fakes = netG(batch["z"], sent_embeddings) 
        fake_kwargs = {"image":fakes}
        if 'cap' in self.g_loss_component or self.fa_feature == "projected_visual_features":
            fake_kwargs.update(
                caption_tokens=batch["caption_tokens"], caption_lengths=batch["caption_lengths"], noitpac_tokens=batch["noitpac_tokens"]
            )
        fake_dict = netD(**fake_kwargs)

        if 'logit' in self.g_loss_component:
            fake_output = netD.logitor(fake_dict[self.logit_input], sent_embeddings)
            if self.type == "hinge":
                errG_fake = -fake_output.mean() 
            else:
                raise NotImplementedError
            loss.update(errG_fake = errG_fake)

        if 'cap' in self.g_loss_component:
            errG_cap = fake_dict["cap_loss"]
            loss.update(errG_cap = errG_cap)
            
        if 'fa' in self.g_loss_component:
            kwargs = {"image":batch["image"]}
            if self.fa_feature == "projected_visual_features":
                kwargs.update(
                    caption_tokens=batch["caption_tokens"], caption_lengths=batch["caption_lengths"], noitpac_tokens=batch["noitpac_tokens"]
                )
            with torch.no_grad():
                real_dict = netD(**kwargs)
            errG_fa = torch.abs(real_dict[self.fa_feature] - fake_dict[self.fa_feature]).mean()
            loss.update(errG_fa = errG_fa)

        return loss, fakes

    def accumulate_loss(self, loss_dict):
        loss = 0.
        for key in loss_dict:
            loss += loss_dict[key] 
        return loss

   







