# Licence: crack_presence_classifier

- **Use:** commercial use allowed (MIT weights, CC BY 4.0 data with attribution).
- **Weights:** MIT. Full text: `LICENSES/crack_presence_classifier.txt`.
- **Base model:** torchvision ResNet18, ImageNet-1K weights; torchvision is BSD 3-Clause
  (https://github.com/pytorch/vision/blob/main/LICENSE).

## Training data

| Dataset | Licence | Attribution |
|---|---|---|
| SDNET2018 (Utah State University) | CC BY 4.0 (https://digitalcommons.usu.edu/all_datasets/48/) | Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning applications. Utah State University. https://doi.org/10.15142/T3TD19 |
| Concrete Crack Images for Classification (Kaggle mirror `arunrk7/surface-crack-detection`) | CC BY 4.0 (https://data.mendeley.com/datasets/5y9wdsg2zt/2) | Özgenel, Ç.F., Gönenç Sorguç, A. "Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings", ISARC 2018, Berlin. |

## Caveat

torchvision's documentation says: "The pre-trained models provided in this library may
have their own licenses or terms and conditions derived from the dataset used for
training." The ImageNet terms restrict the *images* to "non-commercial research and
educational purposes" (https://image-net.org/download.php). Whether those terms reach
weights initialised from an ImageNet-trained backbone is not settled. This was not treated
as a blocker; a commercial user with a low tolerance for risk should make their own
assessment.
