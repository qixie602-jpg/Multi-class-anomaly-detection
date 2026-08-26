# Dataset Descriptions for CPR-MIAD

The datasets are not redistributed in this repository. Download them from their respective original repositories and comply with the corresponding licenses and access conditions.

## Datasets used in CPR-MIAD

### MVTec AD

- Download and extract [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) into `data/mvtec`.
- Run `python data/gen_benchmark/mvtec.py` to obtain `data/mvtec/meta.json` for the standard `DefaultAD` loader in `data/ad_dataset.py`.

```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            └── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        └── ground_truth
            ├── anomaly1
                ├── 000.png
```

### VisA

- Download and extract [VisA](https://registry.opendata.aws/visa/) into `data/visa`.
- Refer to the [project page](https://github.com/amazon-science/spot-diff#data-preparation) for data preparation.
- Run `python data/gen_benchmark/visa.py` to obtain `data/visa/meta.json` for the standard `DefaultAD` loader in `data/ad_dataset.py`.

### MVTec LOCO AD

- Download [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) from its official dataset page and extract it into `data/mvtec_loco`.
- Refer to the same official page for the evaluation protocol, license, and citation information.

### BTAD

- Download [BTAD](https://datasetninja.com/btad) and extract it into `data/btad`.
- The original download server is currently unreliable. Refer to the [official VT-ADL repository](https://github.com/pankajmishra000/VT-ADL#beantech-anomaly-detection-dataset---btad) for provenance and citation information.

### MPDD

- Download [MPDD](https://github.com/stepanje/MPDD) from the link provided by the dataset authors and extract it into `data/mpdd`.
- Refer to the same official repository for the dataset description, license, and citation information.

### Real-IAD

- Request access to and download [Real-IAD](https://huggingface.co/datasets/Real-IAD/Real-IAD), then extract it into `data/realiad`.
- Refer to the [official project page](https://realiad4ad.github.io/Real-IAD/) for dataset details and citation information.
