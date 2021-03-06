import copy
import functools
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from capD.data.tokenizers import SentencePieceBPETokenizer
from capD.modules.textual_heads import TextualHead
from capD.modules.visual_backbones import VisualBackbone
from capD.modules.logitor_backbones import LogitorBackbone

class DiscriminatorModel(nn.Module):
    def __init__(
        self,
        visual: VisualBackbone,
        logitor: LogitorBackbone,
        img_decoder: Any,
    ):
        super().__init__()
        self.visual = visual
        self.logitor = logitor
        self.img_decoder = img_decoder

    def forward(self, image: torch.Tensor, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class MLPMatchingTextualHead(nn.Module):
    def __init__(self, ):
        super().__init__()
        #TODO
        visual_feature_size = 512
        hidden_size = 256 #damsm 

        self.output = nn.Sequential(
            nn.Linear(visual_feature_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Project visual features directly to predict a distribution over
        vocabulary tokens through a single linear layer. This textual head
        ignores arguments ``caption_tokens`` and ``caption_lengths``, they
        are here for API consistency.
        Args:
            visual_features: A tensor of shape ``(batch_size, channels, height,
                width)`` containing features from visual backbone.
        Returns:
            A tensor of shape ``(batch_size, vocab_size)`` containing output
            vocabulary logits.
        """

        # Convert to NHWC and project visual features to textual feature size.
        batch_size, channels, _, _ = visual_features.size()
        visual_features = visual_features.view(batch_size, channels, -1)
        visual_features = visual_features.permute(0, 2, 1)

        # Perform global average pooling of visual features.
        # shape: (batch_size, channels)
        visual_features = visual_features.mean(dim=1)

        # shape: (batch_size, 256)
        proj_features = self.output(visual_features)
        return proj_features 

class MAT_UNCOND(DiscriminatorModel):
    def __init__(
        self,
        visual: VisualBackbone,
        logitor: LogitorBackbone,
        img_decoder: Any = None,
    ):
        super().__init__(visual, logitor, img_decoder)
        self.output = MLPMatchingTextualHead()


    def forward(
        self,
        image: torch.Tensor,
        sent_embs: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, Any]:
        output_dict = {}
        labels = torch.diag(torch.ones(image.size(0))).cuda().detach()

        # shape: (batch_size, channels, height, width)
        logit_features, dec_features, visual_features = self.visual(image, return_features=True)

        output_dict["logit_features"] = logit_features
        output_dict["dec_features"] = dec_features
        output_dict["visual_features"] = visual_features

        if sent_embs is not None:
            img_embs = self.output(visual_features)
            sent_norm = F.normalize(sent_embs, p=2, dim=1)
            img_norm = F.normalize(img_embs, p=2, dim=1)

            scores = torch.mm(sent_norm, img_norm.T) * 10

            s0 = F.log_softmax(scores, dim=0)
            s0 = s0 * labels
            s0 = - (s0.sum(0))
            s0 = s0.mean()

            s1 = F.log_softmax(scores, dim=1)
            s1 = s1 * labels
            s1 = - (s1.sum(1))
            s1 = s1.mean()

            loss = s0 + s1
            output_dict["mat_loss"] = loss
        return output_dict

class DF_D(DiscriminatorModel):
    def __init__(
        self,
        visual: VisualBackbone,
        logitor: LogitorBackbone,
        img_decoder: Any = None,
    ):
        super().__init__(visual, logitor, img_decoder)

    def forward(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        output_dict = {}
        # shape: (batch_size, channels, height, width)
        logit_features, dec_features, visual_features = self.visual(image, return_features=True)

        output_dict["logit_features"] = logit_features
        output_dict["dec_features"] = dec_features
        output_dict["visual_features"] = visual_features
        return output_dict

class CapD(DiscriminatorModel):
    def __init__(
        self,
        visual: VisualBackbone,
        logitor: LogitorBackbone,
        textual: TextualHead,
        caption_backward: bool = True,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
        img_decoder: Any=None,
    ):
        super().__init__(visual, logitor, img_decoder)
        self.textual = textual
        self.padding_idx = self.textual.padding_idx
        self.caption_backward = caption_backward

        if self.caption_backward:
            self.backward_textual = copy.deepcopy(self.textual)
            self.backward_textual.visual_projection = self.textual.visual_projection
            self.backward_textual.embedding = self.textual.embedding
            self.backward_textual.output = self.textual.output

        self.sos_index = sos_index
        self.eos_index = eos_index
        self.decoder = decoder
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

    def forward(
        self, 
        image: torch.Tensor, 
        batch: Dict[str, torch.Tensor] = {},
        cap_stop_grad: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:

        # Compute features and captioning loss                
        output_dict = {}
        # shape: (batch_size, channels, height, width)
        logit_features, dec_features, visual_features = self.visual(image, return_features=True)
        batch_size = logit_features.size(0)

        output_dict["logit_features"] = logit_features
        output_dict["visual_features"] = visual_features
        output_dict["dec_features"] = dec_features

        if "caption_tokens" in batch:
            caption_tokens = batch["caption_tokens"]
            caption_lengths = batch["caption_lengths"]

            # shape: (batch_size, max_caption_length, vocab_size)
            if cap_stop_grad:
                visual_features = visual_features.detach()
                
            output_logits, projected_visual_features = self.textual(
                visual_features, caption_tokens, caption_lengths
            )
            loss = self.loss(
                output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                caption_tokens[:, 1:].contiguous().view(-1),
            )
            output_dict["cap_loss"] = loss
            output_dict["projected_visual_features"] = projected_visual_features

            # Do captioning in backward direction if specified.
            if self.caption_backward:
                backward_caption_tokens = batch["noitpac_tokens"]

                backward_output_logits, _ = self.backward_textual(
                    visual_features, backward_caption_tokens, caption_lengths
                )
                backward_loss = self.loss(
                    backward_output_logits[:, :-1]
                    .contiguous()
                    .view(-1, self.textual.vocab_size),
                    backward_caption_tokens[:, 1:].contiguous().view(-1),
                )
                output_dict["cap_loss"] += backward_loss

            output_dict["predictions"] = torch.argmax(output_logits, dim=-1)
        elif not self.training:
            if self.decoder is None:
                raise ValueError("Decoder for predicting captions is missing!")

            # During inference, get beam search predictions for forward
            # model. Predictions from forward transformer will be shifted
            # right by one timestep.
            start_predictions = visual_features.new_full(
                (batch_size,), self.sos_index
            ).long()
            # Add image features as a default argument to match callable
            # signature accepted by beam search class (partial captions only).
            decoding_step = functools.partial(self.decoding_step, visual_features)

            predicted_caption, _ = self.decoder.search(
                start_predictions, decoding_step
            )
            output_dict = {"predictions": predicted_caption}

        return output_dict


    def decoding_step(
        self, visual_features: torch.Tensor, partial_captions: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Given visual features and a batch of (assumed) partial captions, predict
        the logits over output vocabulary tokens for next timestep. This method
        is used by :class:`~capD.utils.beam_search.AutoRegressiveBeamSearch`
        and :class:`~capD.utils.nucleus_sampling.AutoRegressiveNucleusSampling`.

        .. note::

            For nucleus sampling, ``beam_size`` will always be 1 (not relevant).

        Parameters
        ----------
        projected_visual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., textual_feature_size)``
            with visual features already projected to ``textual_feature_size``.
        partial_captions: torch.Tensor
            A tensor of shape ``(batch_size * beam_size, timesteps)``
            containing tokens predicted so far -- one for each beam. We need all
            prior predictions because our model is auto-regressive.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size * beam_size, vocab_size)`` -- logits
            over output vocabulary tokens for next timestep.
        """

        # Expand and repeat image features while doing beam search.
        batch_size, channels, height, width = visual_features.size()
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, channels, height, width
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a timestep. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        logits, _ = self.textual(visual_features, partial_captions, caption_lengths)
        # Return logits from the last timestep.
        return logits[:, -1, :]

    def log_predictions(
        self, batch: Dict[str, torch.Tensor], tokenizer: SentencePieceBPETokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            predictions_str += f"""
                Caption tokens : {" ".join(tokens.tolist())}
                Predictions (f): {" ".join(preds.tolist())}

                """
        return predictions_str