# SkyDiffusion 作者材料请求草稿

日期：2026-10-13。状态：**仅草稿，尚未发送**。发送前请检查内容并补充姓名/单位；未声称已获得作者材料。

## 英文邮件

To: `yejy53@mail2.sysu.edu.cn`

Subject: SkyDiffusion ICCV 2025 — request for the G2A-3 Disaster reproducibility package

Dear Junyan Ye and the SkyDiffusion team,

Thank you for sharing SkyDiffusion and the G2A-3 dataset. I am working on reproducing the Disaster experiment reported in Table 3 of your ICCV 2025 paper, “Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis.”

I have run the public implementation at commit `461ccc2fff11882d48f5ab62cae65eef5c45c647` with the publicly linked `CVACT_Skydiffusion.ckpt`. Its SHA256 is `a9108a3394fe127ae7cde9a75019b34ea11da9d3375a51efb4236200bbedfc6a`, and all model keys and tensor shapes load strictly. I understand that this CVACT checkpoint should not be assumed to reproduce the G2A-3 Disaster results.

Could you please point me to the following materials, or let me know whether they can be shared?

1. **Task checkpoint and configuration.** The checkpoint used for the Disaster result in Table 3, its checksum, the training/inference configuration, and the exact code commit. If the checkpoint cannot be released, would it be possible to share the generated images for the official Disaster test set?
2. **Official data split.** Disaster train/test sample IDs or manifests, the regional split boundaries, and the dataset revision used in the paper. The paper specifies a regional 4:1 split, but I could not locate the actual IDs in the public materials I checked.
3. **Projection/preprocessing.** Disaster meters per pixel, signed camera height, one-to-one curve coefficient λ, panorama resize dimensions, and north/vertical coordinate conventions. The public code defaults to 0.06 m/pixel, −1.5 m and a curve coefficient of 3; I do not want to assume that these were the Disaster training settings. Also, the paper states a 512×1024 panorama, while the public CVACT branch uses 832×1664.
4. **Evaluation protocol.** The exact commands and dependency versions for PSNR, SSIM, LPIPS and FID. In particular, `eval/evaluation.py` compares predictions with `*_reconstruction.png` produced by the VAE; should the paper metrics use the original aerial targets or their VAE reconstructions? Please also clarify SSIM channel/window, LPIPS network/version, PNG quantization, FID implementation/Inception weights, and the reference statistics or image list.

As a diagnostic only, I evaluated 8 Disaster pairs using the CVACT checkpoint (50-step DDIM, guidance parameter 9, empty positive/negative prompts, FP32). Against original targets, the means were PSNR 12.136 dB, Y-SSIM 0.1045, and AlexNet LPIPS 0.7120. These are not official-split results or a claim about your method's benchmark performance. Evaluating the same predictions against separately produced VAE target reconstructions changes the scores, so I would appreciate clarification of the intended reference.

I can provide a concise reproducibility log if helpful. My immediate priority is Disaster; if the corresponding UAV/History packages are also available, their links would be greatly appreciated.

Thank you for your time and help.

Best regards,

[Your name]

[Your affiliation, optional]

## GitHub Issue 草稿

发布入口：[官方仓库 Issues](https://github.com/opendatalab/skydiffusion/issues)。建议发布前先查看现有权重相关 Issue，避免重复；也可以把下列内容作为现有讨论的补充。

Title: G2A-3 Disaster reproduction: task checkpoint, regional split, projection settings and evaluation reference

Thank you for releasing SkyDiffusion. I am trying to reproduce the G2A-3 Disaster result in ICCV 2025 Table 3, rather than only run a demo.

The public CVACT checkpoint loads strictly with commit `461ccc2fff11882d48f5ab62cae65eef5c45c647`. However, I could not locate the following materials in the public sources I checked:

- The Disaster checkpoint and its training/inference configuration/code revision.
- Official Disaster train/test sample IDs and the regional boundaries for the 4:1 split.
- Disaster-specific meters per pixel, signed camera height, curve λ, panorama dimensions and orientation conventions.
- A complete evaluation command/version list, particularly whether the reference is the original target or the VAE reconstruction used in `eval/evaluation.py`, and the FID implementation/reference statistics.

Could you please provide links or clarify which public configuration reproduces this table row? If the weights cannot be released, official test predictions and the evaluation protocol would also be useful.

I am not treating the CVACT checkpoint's small-sample cross-domain scores as a formal reproduction result. Thank you for any guidance.
