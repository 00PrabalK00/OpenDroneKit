# Licence: structural_multiclass_detector

- **Use:** non-commercial only (CODEBRIM terms), and the weights are also AGPL-3.0
  (an Ultralytics YOLO11x fine-tune). See the OWNER GATE in `WEIGHTS_LICENSES.md`:
  AGPL-3.0 forbids adding restrictions such as "non-commercial", so no single set of terms
  satisfies both. Shipped flagged, with both notices, pending the owner's decision.
- **Weights:** AGPL-3.0 (Ultralytics, https://www.ultralytics.com/license) and the CODEBRIM
  licence agreement. Full texts: `LICENSES/structural_multiclass_detector.txt`.
- **Decision under the owner's rule:** ship. The CODEBRIM licence does not forbid
  distributing trained models; clause 5 expressly permits sharing them non-commercially.

## Training data

CODEBRIM (COncrete DEfect BRidge IMage Dataset), Goethe University Frankfurt and FIAS. The
model was trained on the Roboflow export `codebrim_structural`, whose "CC BY 4.0" label
cannot relicense the original. The original's terms govern: Zenodo record 2620293
(`license.id = other-nc`) and its `license.md`
(https://github.com/MrtnMndt/meta-learning-CODEBRIM/blob/master/license.md).

## Clauses quoted (CODEBRIM license.md)

> 2. The researcher shall use the database and meta-learning code for non-commercial
> research and educational purposes only. ...
>
> 3. The researcher may not use modified versions of the dataset, code or any derivative
> works (such as additional annotations, trained models or anything that directly includes
> any of the data) to procure a commercial gain.
>
> 4. The researcher shall not distribute the database or modified versions thereof. ...
>
> 5. Derivative works such as additional annotations or trained models that do not
> directly include any of the data are permissible to be shared under the same license
> agreement for non-commercial and educational purposes.
>
> 6. The researcher shall include a reference to the CODEBRIM database and its
> corresponding publication ...

The ONNX file holds network weights only. It does not include any CODEBRIM image.

## Attribution

- Martin Mundt, Sagnik Majumder, Sreenivas Murali, Panagiotis Panetsos and Visvanathan
  Ramesh. "Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect
  Classification with the COncrete DEfect BRidge IMage Dataset". Conference on Computer
  Vision and Pattern Recognition (CVPR), 2019.
- Ultralytics YOLO11 (https://github.com/ultralytics/ultralytics), AGPL-3.0.
