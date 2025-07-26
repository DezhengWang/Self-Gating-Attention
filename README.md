<div align="center">
  
# Self-Gating-Attention
Self-Gating Attention for Forecasting Time Series

<img src="https://github.com/DezhengWang/Self-Gating-Attention/blob/main/alpha_v.png" alt="Self-gating attention mechanism." width="200" />

</div>

Unlike standard self-attention, Self-Gating-Attention (SGA) does not rely on query-key similarity. Instead, it computes attention scores directly from the input. This simplified architecture improves computational efficiency while preserving competitive predictive performance, particularly in long-range forecasting scenarios. The highlights are as follows:

1) We observe that similar attention score patterns appear across different sampling times in time series forecasting models. By relying only on a static and shared attention score, the model also achieves relatively competitive performance compared with self-attention.

2) We introduce the SGA as an efficient alternative to canonical self-attention. SGA eliminates the reliance on query-key similarity, thereby reducing the number of projection operations by two-thirds before input to the attention component, which achieving linear time and memory complexity, i.e., $O(n)$ with respect to sequence length $n$.
