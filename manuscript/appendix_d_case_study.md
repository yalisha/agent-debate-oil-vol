# Appendix D: Crisis Episode Case Study (March 2022)

The Russia-Ukraine conflict escalation in late February 2022 triggered the sharpest WTI price spike in the sample period. This appendix examines four trading days during and after the invasion to illustrate how the DropEdge GAT corrects debate forecasts under crisis conditions.

## D.1 Attribution Dynamics

Table D1 reports the debate and GAT predictions alongside the actual 20-day forward realised volatility for four dates in March 2022.

**Table D1: GAT correction during the March 2022 crisis**

| Date | Actual vol | Debate pred | GAT pred | HAR | Debate AE | GAT AE | Helped | $n_{\text{herd}}$ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2022-03-09 | 0.704 | 0.767 | 0.583 | 0.509 | 0.064 | 0.121 | No | 1 |
| 2022-03-17 | 0.622 | 0.868 | 0.739 | 0.793 | 0.246 | 0.117 | Yes | 4 |
| 2022-03-28 | 0.547 | 0.934 | 0.802 | 0.861 | 0.387 | 0.255 | Yes | 3 |
| 2022-03-31 | 0.470 | 0.908 | 0.779 | 0.844 | 0.438 | 0.310 | Yes | 3 |

On 9 March, just after the invasion peak, the debate forecast overshoots only slightly (AE $= 0.064$) and the GAT's correction toward a lower prediction is unhelpful. By 17 March, the debate substantially overshoots (AE $= 0.246$) as agents persist with crisis-level adjustments even as realised volatility begins to decline. The GAT pulls the forecast down, cutting the error roughly in half. This pattern intensifies on 28 and 31 March: the debate rule continues to predict near-unit volatility while actual volatility has fallen to 0.47-0.55, and the GAT reduces the absolute error by 34-39 percent in each case.

## D.2 Herding Dynamics

The herding count rises from 1 on 9 March (low herding, agents still processing the invasion independently) to 3-4 on 17-31 March (most agents anchored on the elevated consensus from the invasion week). The GAT node features encode this herding shift, and the regime gate ($g(t)$) increases during this period as persistence volatility remains elevated, shifting weight toward the regime-specialised head.

## D.3 Interpretation

The case study illustrates two properties of the framework. When the debate consensus is close to the realised outcome (9 March), the GAT's correction provides no benefit and can slightly worsen the forecast; the model is not uniformly superior. When agents herd on an outdated consensus while market conditions normalise (17-31 March), the learned combination detects the herding pattern through the behavioural features and corrects toward the regime-appropriate prediction. This asymmetry between the "anchored debate" failure mode and the "diverse debate" success mode is what the learned sparse graph is designed to exploit.
