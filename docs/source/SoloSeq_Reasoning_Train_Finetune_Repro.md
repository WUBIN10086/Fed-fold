<!--
该文档用于记录 SoloSeq 的推理流程、训练流程、微调流程复现。
先保持为空白，后续逐步补全。
-->
# SoloSeq推理，训练，微调全流程记录
本文档使用SoloSeq 路线：不依赖 MSA，直接做单序列训练

## 0. Inital
这条线先下载 `openfold_soloseq_params`，然后从 `pdb_mmcif/` 里取一个子集 `train_subset/`，接着先生成 FASTA，再生成 `embedding`，最后生成 `cache`，并用 `--use_single_seq_mode True` 启动训练，配置是 `seq_model_esm1b_ptm`.

推荐使用conda env环境（个人习惯哈哈）

因为官方只给了一个yml的环境文件，我添加了一个requirements方便安装环境。注意torch和cuda需要根据显卡。

---

## 完整 SoloSeq 推理流程

**第 1 步：准备 FASTA 输入**

官方要求是：`examples/monomer/fasta_dir/` 目录下放 FASTA 文件，通常 一个文件一条序列:
```
fasta_dir/
└── protein1.fasta
```
*根据Gemini的回答，最常用的蛋白质预测练手对象：`Hen Egg-White Lysozyme`,`Myoglobin`,`Ubiquitin`*

这里首先选用`Ubiquitin(P62068)`进行推理测试。FASTA下载地址为：https://www.uniprot.org/.

**第 2 步：下载 SoloSeq 权重**

运行命令下载官方权重.

注意: 需要aws工具，请先在linux中安装。
```bash
sudo apt update
sudo apt install awscli
```
若系统里 apt 没有可用的 awscli 包，可以用官方方式安装 AWS CLI v2

```bash
cd /tmp
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip
sudo ./aws/install
aws --version
```

下载权重 (1.1GB)
```bash
bash scripts/download_openfold_soloseq_params.sh openfold/resources
```

**第 3 步：预计算 ESM-1b embedding**

soloSeq实际上使用了ESM-1b模型进行MAS，这里需要运行python以及torch环境，需要根据显卡进行选择。

推荐使用虚拟环境（我用的miniconda）
```bash
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

运行conda环境（python推荐3.10）
```bash
conda create -n Fedfold python=3.10
conda activate Fedfold
pip install -r requirements.txt
```

这里我使用的工作站显卡A4500, 推荐cuda版本12.0，请GPT一下自己的显卡对应的cuda以及torch版本。另外sm120架构的新版blackwell显卡们对应的torch好像已经解决了。

在conda环境下安装cuda和torch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

ESM-1b embedding:
```bash
python scripts/precompute_embeddings.py examples/monomer/fasta_dir/ embeddings_output_dir/
```

这个地方我遇到了`ModuleNotFoundError: No module named 'openfold'`问题，问了一下GPT老师：
```bash
cd ~/Desktop/Fed-fold
export PYTHONPATH=$(pwd):$PYTHONPATH
python -c "import torch; print(torch.__version__)"
python -c "import openfold; print(openfold.__file__)"
```

再次运行precompute_embedding成功

**第 4 步（可选）：准备模板 `.hhr`**

官方 `README `说明，在`embeddings_output_dir` 的每个样本子目录里，还可以放 `*.hhr` 文件。

这个 `.hhr` 是 `HHSearch` 输出，表示模板检索结果

+ 有 .hhr：模型会同时使用 embedding + template

+ 没有 .hhr：模型只使用 ESM-1b embedding

**第 5 步：执行 SoloSeq 推理**

官方文档入口：
```bash
python run_pretrained_openfold.py \
    examples/monomer/fasta_dir \
    data/pdb_mmcif/mmcif_files/ \
    --use_precomputed_alignments embeddings_output_dir \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
```

