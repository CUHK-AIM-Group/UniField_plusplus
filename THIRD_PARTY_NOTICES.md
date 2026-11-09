# Third-party sources

The bundled DiffSynth/FlashVSR-derived files retain the repository's Apache-2.0 license and existing notices. External packages retain their own licenses.

* FlashVSR: https://github.com/OpenImagingLab/FlashVSR and https://huggingface.co/JunhaoZhuang/FlashVSR (Apache-2.0).
* DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio (Apache-2.0).
* Wan2.1: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B.
* Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B.
* Block-Sparse-Attention: https://github.com/mit-han-lab/Block-Sparse-Attention (BSD-3-Clause).
* Harvard-Oxford: distributed by FSL/NITRC; https://fsl.fmrib.ox.ac.uk/fsl/docs/#/other/datasets and https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz. The downloader fetches the source atlas; the code license does not relicense this data.
* MNI152NLin6Asym templates: https://github.com/templateflow/tpl-MNI152NLin6Asym and https://www.templateflow.org/.
* ANTsPy: https://github.com/ANTsX/ANTsPy.
* MONAI: https://github.com/Project-MONAI/MONAI.
* LPIPS: https://github.com/richzhang/PerceptualSimilarity. Its AlexNet backbone is downloaded from the official PyTorch model host.

The initialization assets contain generator Stage-1 LoRA and planner LoRA parameters; their respective base weights must be obtained from the sources in the manifest. They contain no training data, optimizer states or completed main/decoder experimental outputs.
