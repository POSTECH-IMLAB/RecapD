import copy
import functools
from typing import Any, Dict

import torch
from torch import nn

from capD.data.tokenizers import SentencePieceBPETokenizer
from capD.modules.textual_heads import TextualHead
from capD.modules.visual_backbones import VisualBackbone
from capD.modules.logitor_backbones import LogitorBackbone

class DiscriminatorModel(nn.Module):
    def __init__(
        self,
        visual: VisualBackbone,
        logitor: LogitorBackbone,
    ):
        super().__init__()
        self.visual = visual
        self.logitor = logitor

    def forward(self, image: torch.Tensor, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class DF_D(DiscriminatorModel):
    def __init__(
        self,
        visual: VisualBackbone,
        logitor: LogitorBackbone,
    ):
        super().__init__(visual, logitor)

    def forward(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        output_dict = {}
        # shape: (batch_size, channels, height, width)
        visual_features = self.visual(image)
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
        super().__init__(visual, logitor)
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
        self.img_decoder = img_decoder
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

    def forward(
        self, 
        image: torch.Tensor, 
        batch: Dict[str, torch.Tensor] = {}, 
    ) -> Dict[str, Any]:

        # Compute features and captioning loss                
        output_dict = {}
        # shape: (batch_size, channels, height, width)
        visual_features = self.visual(image)
        batch_size = visual_features.size(0)

        output_dict["visual_features"] = visual_features

        if "caption_tokens" in batch:
            caption_tokens = batch["caption_tokens"]
            caption_lengths = batch["caption_lengths"]

            # shape: (batch_size, max_caption_length, vocab_size)
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
        logits = self.textual(visual_features, partial_captions, caption_lengths)
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