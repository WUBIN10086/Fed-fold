![header ](imgs/of_banner.png)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._

# FedFold 重写版文档

FedFold 是一个面向蛋白质结构预测的联邦学习 SoloSeq 微调仓库，以 5 个 client
本地微调、FedAvg 全局聚合和独立测试集评测为核心流程，支持从数据准备、pLDDT
难样本筛选、train/test 划分到 TM-score 评测矩阵的完整实验闭环。

本仓库主要参考了 [OpenFold](https://github.com/aqlaboratory/openfold) 和
[AlphaFold 2](https://github.com/deepmind/alphafold)。

# 使用方法：

请参考Sha_log[pipeline](docs/sha_log/fed_finetune_pipeline.md)

# 任务进度：
## WU:
1. 下载了从2025-01-01到2026-07-01的数据(符合结构的总数4334)
2. 修改了一下data下载的文件夹名字，对应需要修改的路径也一起改了(一年的数据在data/all_pdb_1y)
3. data_dir_to_fasta脚本进行了修改，避免一次写入(我的内存不够ORZ)
4. 数据聚类划分已提交，5个client，mmcif文件没有上传，上传了处理好的fasta序列。可以按照姓氏排名分配：
+ He -> client 0
+ HU -> client 1
+ Sha -> client 2
+ Wu -> client 3
+ Wang -> client 4

# openfold原始文档引导

See our new home for docs at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/), with instructions for installation and model inference/training.

Much of the content from this page may be found [here.](https://github.com/aqlaboratory/openfold/blob/main/docs/source/original_readme.md)

## Copyright Notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters
fall under the CC BY 4.0 license, a copy of which is downloaded to
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.

## Contributing

If you encounter problems using OpenFold, feel free to create an issue! We also
welcome pull requests from the community.

## Citing this Work

Please cite our paper:

```bibtex
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

If you use OpenProteinSet, please also cite:

```bibtex
@misc{ahdritz2023openproteinset,
      title={{O}pen{P}rotein{S}et: {T}raining data for structural biology at scale}, 
      author={Gustaf Ahdritz and Nazim Bouatta and Sachin Kadyan and Lukas Jarosch and Daniel Berenberg and Ian Fisk and Andrew M. Watkins and Stephen Ra and Richard Bonneau and Mohammed AlQuraishi},
      year={2023},
      eprint={2308.05326},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

Any work that cites OpenFold should also cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.
