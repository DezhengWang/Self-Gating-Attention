<div align="center">
  
# Self-Gating-Attention
Self-Gating Attention for Forecasting Time Series

<img src="https://github.com/DezhengWang/Self-Gating-Attention/blob/main/alpha_v.png" alt="Self-gating attention mechanism. The symbol ‘?’ indicates whether sparsity is required." width="200" />
</div>

Unlike standard self-attention, SGA does not rely on query-key similarity. Instead, it computes attention scores directly from the input. This simplified architecture improves computational efficiency while preserving competitive predictive performance, particularly in long-range forecasting scenarios. The highlights are as follows:

1) the SGA serves as an efficient alternative to canonical self-attention. It achieves linear time and memory complexity, i.e., $O(n)$ with respect to sequence length $n$, while preserving effective dependency alignment.

2) SGA eliminates the reliance on query-key similarity, thereby reducing the number of projection operations by two-thirds before input to the attention component.
