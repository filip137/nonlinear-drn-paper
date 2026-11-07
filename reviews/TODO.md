# Reviews to do

## 8 November 2026

- [ ] Revise the passage introducing nonlinearity and Figure 3a–b. Start by motivating the analysis: we are explaining the **number of iterations needed to reach equilibrium**, following the discussion of cost per iteration. Explain why we examine diode conduction and current tolerance, then introduce the metric and results. Consider focusing this discussion on nonlinear networks only; this is an open editorial choice.

Passage to revisit: “We quantify nonlinearity by the fraction of conducting hidden nodes, averaged over all inputs” through the Figure 3b tolerance comparison. Define a conducting node by $g_{\mathrm d}=\mathrm{d}i_{\mathrm d}/\mathrm{d}v\geq0.1g_{\mathrm{lin}}$, with $g_{\mathrm{lin}}$ the sum of linear conductances connected to that node (correct “note” to “node”).

Keep the current results available for the rewrite: Figure 3a varies input gain at fixed weights; increasing conduction generally increases Newton steps and decreases CD sweeps. The proposed explanation is that diode conduction increases the Jacobian diagonal while interlayer couplings stay fixed, which can improve CD convergence. Figure 3b fixes gain at 30.8% conducting hidden nodes; tightening current tolerance from $10^{-3}$ to $10^{-7}$ A changes mean CD sweeps from 16.2 to 24.7 and Newton steps from 18.7 to 19.8. Connect these observations explicitly to the iteration-count question.

- [ ] Revise the Figure 3 caption to match the revised iteration-count discussion and the rest of the paper. Keep terminology, notation, panel descriptions, and the distinction between Newton steps and CD sweeps consistent with the main text and other figure captions.

- [ ] Make the R1.4 response much more concise. Probably omit the second paragraph; review whether any essential point from it needs to be retained in the shortened reply.

- [ ] Standardize SPICE/ELDO naming and capitalization across all figures, including legends, axis labels, annotations, and captions. Use a consistent convention aligned with the main text.
