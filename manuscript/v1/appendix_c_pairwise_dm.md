# Appendix C: Pairwise HAC-Corrected Diebold-Mariano Statistics

Table C1 reports pairwise HAC-corrected Diebold-Mariano statistics for the principal models. Each cell shows $\text{DM}_{\text{HAC}}$ (row model vs column model), with the $p$-value in parentheses. Negative values indicate that the row model has lower squared error than the column model. All tests use Newey-West HAC standard errors with Bartlett kernel and bandwidth 19.

**Table C1: Pairwise DM$_{\text{HAC}}$ statistics ($n = 972$)**

| | HAR | GARCH | MLP | Debate | Single |
|---|:---:|:---:|:---:|:---:|:---:|
| **DropEdge GAT** | $-$3.212 (0.001)\*\*\* | $-$1.948 (0.051) | $-$3.336 (<0.001)\*\*\* | | |
| **Learned GAT** | $-$3.165 (0.002)\*\* | | | | |
| **GARCH** | $-$2.604 (0.009)\*\* | --- | | | |
| **MLP** | $-$1.214 (0.225) | | --- | | |
| **Debate** | $-$1.494 (0.135) | | | --- | |
| **Single agent** | $-$1.230 (0.219) | | | | --- |

Additional pairwise comparisons from the ablation analysis:

| Comparison | DM$_{\text{HAC}}$ | $p$ |
|---|:---:|:---:|
| DropEdge GAT vs Dense GAT | $-$6.382 | $<$0.001\*\*\* |
| DropEdge GAT vs Identity | $-$1.107 | 0.268 |
| Dense GAT vs Identity | 6.365 | $<$0.001\*\*\* |

The DropEdge GAT significantly outperforms the MLP ($p < 0.001$) and the Dense GAT ($p < 0.001$), and marginally outperforms GARCH at the 10 percent level ($p = 0.051$). The Identity variant (self-loops only) significantly outperforms the Dense GAT ($p < 0.001$), confirming that full connectivity is harmful and that learned sparsity is the source of the graph structure's contribution.
