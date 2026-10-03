# Licence: crack_segmentation

- **Use:** non-commercial only: research or evaluation.
- **Weights:** NVIDIA Source Code License for SegFormer
  (https://github.com/NVlabs/SegFormer/blob/master/LICENSE), because they are a fine-tune of
  `nvidia/mit-b5`. Full text: `LICENSES/crack_segmentation.txt`.
- **Redistribution:** allowed under the same licence (section 3.1); use is limited to
  research or evaluation (section 3.3).
- **Decision under the owner's rule:** ship, non-commercial. Every data source was read
  (below). Several restrict use to non-commercial research, and GAPs forbids distributing
  *the database*. None forbids distributing a model trained on it, so none is an outright
  ban on redistributing these weights.

## Training data: each source's terms, as read on 2026-10-03

Corpus `crack_seg` = `crack_segmentation_kaggle` + `deepcrack` + `crackforest`
(`training/datasets/prepare.py`). The Kaggle set aggregates public crack sets, named by
file prefix.

| Source | Terms found | Clause | Bars distributing trained weights? |
|---|---|---|---|
| Kaggle aggregate `lakshaymiddha/crack-segmentation-dataset` (from `khanhha/crack_segmentation`) | Kaggle licence field "Unknown"; the GitHub repository has no licence | khanhha README: "please cite the corresponding papers when using these datasets." | No clause found |
| CFD / CrackForest (`cuilimeng/CrackForest-dataset`) | Non-commercial research | README: "The dataset is made available for non-commercial research purposes only." | No |
| DeepCrack (`yhlleo/DeepCrack`) | Non-commercial research and education | README: "we own the copyrights to part of original crack images and all annotated maps. Their use is RESTRICTED to non-commercial research and educational purposes." | No. The provenance manifest's earlier "MIT (repository)" described the code, not the data. |
| CRACK500 (`fyangneil/pavement-crack-detection`) | No licence stated; the repository has no licence | README: "please cite the corresponding papers when using these datasets." | No clause found |
| GAPs384 (TU Ilmenau GAPs) | Academic use only | Request form, "Terms and Conditions of Usage": "publicly available for academic use only ... Any commercial distribution or act related to the commercial usage of this database is strictly prohibited. The distribution of this database to any parties that have not read and agreed to the terms and conditions of usage is strictly prohibited." | No. The bar is on distributing the database, not trained models; the commercial bar is honoured by the non-commercial release. |
| Rissbilder, Volker (via `alexdonchuk/cracks_segmentation_dataset`) | CC BY 4.0 (repository LICENSE) | Attribution required | No |
| AEL (Amhaz, Chambon et al.) | No licence found; the hosting page (irit.fr/~Sylvie.Chambon) now returns 404 | none found | No clause found |
| CrackTree200 (Zou et al.) | No licence found | none found | No clause found |

Sources with no stated licence are not treated as cleared for anything beyond what their
authors published. The weights are released non-commercially, and no image is
redistributed.

## Attribution

- CRACK500: Zhang, L., Yang, F., Zhang, Y. D., Zhu, Y. J. "Road crack detection using deep
  convolutional neural network", ICIP 2016; Yang, F. et al. "Feature Pyramid and
  Hierarchical Boosting Network for Pavement Crack Detection", IEEE T-ITS 2019.
- GAPs384: Eisenbach, M. et al. "How to Get Pavement Distress Detection Ready for Deep
  Learning? A Systematic Approach", IJCNN 2017; Yang et al. 2019 (above).
- CFD / CrackForest: Shi, Y., Cui, L., Qi, Z., Meng, F., Chen, Z. "Automatic road crack
  detection using random structured forests", IEEE T-ITS 17(12), 2016.
- AEL: Amhaz, R., Chambon, S., Idier, J., Baltazart, V. "Automatic Crack Detection on
  Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection", IEEE
  T-ITS 2016.
- CrackTree200: Zou, Q. et al. "CrackTree: Automatic crack detection from pavement images",
  Pattern Recognition Letters 33(3), 2012.
- DeepCrack: Liu, Y., Yao, J., Lu, X., Xie, R., Li, L. "DeepCrack: A Deep Hierarchical
  Feature Learning Architecture for Crack Segmentation", Neurocomputing 338, 2019.
- Rissbilder, Volker: `alexdonchuk/cracks_segmentation_dataset`, CC BY 4.0.
- Aggregation: `khanhha/crack_segmentation`; Kaggle `lakshaymiddha/crack-segmentation-dataset`.
- SegFormer, NVIDIA, NVIDIA Source Code License for SegFormer.
