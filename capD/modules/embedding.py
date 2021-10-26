import functools

import torch
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WordAndPositionalEmbedding(nn.Module):
    r"""
    A :class:`~torch.nn.Module` for learned word embeddings and position
    embeddings for input tokens. Each token is mapped to a fixed dimensional
    word embedding; and corresponding positional embedding based on its index.
    These are summed together followed by layer normalization and an optional
    dropout.
    Args:
        vocab_size: Size of token vocabulary.
        hidden_size: Size of token embedding vectors.
        dropout: Probability for final dropout applied after layer normalization.
        max_caption_length: Maximum length of input captions; this is used to create a
            fixed positional embedding lookup table.
        padding_idx: Token index of ``[PAD]`` token, word embedding for these tokens
            will be a vector of zeroes (and not trainable).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Get combined word and positional embeddings for input tokens.
        Args:
            tokens: A tensor of shape ``(batch_size, max_caption_length)``
                containing a batch of caption tokens, values in ``[0, vocab_size)``.
        Returns:
            A tensor of shape ``(batch_size, max_caption_length, hidden_size)``
            containing corresponding token embeddings.
        """
        position_indices = self._create_position_indices(tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens: torch.Tensor):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions


class DAMSM(nn.Module):
    def __init__(
        self,
        max_caption_length,
        frozen = True,
        save_dir = 'datasets/DAMSMencoders/coco/text_encoder100.pth',
        
    ):
        super().__init__()
        self.max_caption_length = max_caption_length
        self.vocab_size = 27297 
        self.ninput = 300
        self.dropout = 0.5 
        self.nlayers = 1
        self.bidirectional = True
        self.rnn_type = 'LSTM'
        self.num_directions = 2
        self.nhidden = 256 // self.num_directions 

        self._define_modules()
        self._load_weights(save_dir)
        if frozen:
            self.eval()
            self.requires_grad_(False)

    def _define_modules(self):
        self.encoder = nn.Embedding(self.vocab_size, self.ninput)
        self.drop = nn.Dropout(self.dropout)
        self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers, batch_first=True,
                           dropout=self.dropout,
                           bidirectional=self.bidirectional)

    def _load_weights(self, save_dir):
        self.load_state_dict(torch.load(save_dir, map_location='cpu'))

    def forward(self, caption_tokens, caption_lengths, **kwargs):

        sorted_cap_lens, sorted_idx = caption_lengths.sort(descending=True)
        sorted_caps = caption_tokens[sorted_idx]
        sorted_cap_lens = sorted_cap_lens.tolist()

        sorted_embs = self.drop(self.encoder(sorted_caps))

        sorted_embs = pack_padded_sequence(sorted_embs, sorted_cap_lens, batch_first=True)
        sorted_outputs, sorted_hiddens = self.rnn(sorted_embs)

        sorted_outputs = pad_packed_sequence(sorted_outputs, batch_first=True, total_length=self.n_steps)[0]

        sorted_words_embs = sorted_outputs.transpose(1, 2)

        sorted_sent_embs = sorted_hiddens[0].transpose(0, 1).contiguous()

        sorted_sent_embs = sorted_sent_embs.view(-1, self.nhidden * self.num_directions)

        mask = (caption_tokens == 0)

        words_embs = sorted_words_embs[sorted_idx.argsort()]
        sent_embs = sorted_sent_embs[sorted_idx.argsort()]

        return words_embs, sent_embs, mask

        
