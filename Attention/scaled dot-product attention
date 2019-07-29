import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        Args:
        	q: Size[B, L_q, D_q]
        	k: Size[B, L_k, D_k]
        	v: Size[B, L_v, D_v]
        	scale: 
        	attn_mask: Masking，Size[B, L_q, L_k]

        Returns:
        	 context tensor and attetention tensor
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        if attn_mask:
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# softmax
        attention = self.softmax(attention)
		# dropout
        attention = self.dropout(attention)
        
        context = torch.bmm(attention, v)
        return context, attention
