# SkyDiffusion 作者材料获取进展

核验日期：2026-10-13。本文记录后续公开渠道检索，不代表已联系作者或已得到作者确认。

## 当前回答

可以继续检索并下载作者公开发布的材料，也可以准备精准的材料请求；**目前尚未取得 G2A-3 任务权重、正式划分、任务投影标定与完整论文评估协议，不能保证作者会回复或提供**。

目前没有发送邮件、发布 GitHub Issue 或访问私有资源，也没有取得作者直接回复。不能把联系邮箱、旧版 README、其他 BEV 模型或复现者的推测当作已经获得缺失实验材料。

## 本轮新核验的渠道

| 渠道 | 公开核验结果 | 对四项缺口的影响 |
|---|---|---|
| [作者公开仓库列表](https://github.com/yejy53?tab=repositories) | 找到作者另一个 `yejy53/Skydiffusion` 仓库；当前 main 树只有 5 个条目，主要是 README 与图片 | 没有找到任务 checkpoint、正式 IDs 或完整实验配置 |
| [作者旧版 Skydiffusion 仓库](https://github.com/yejy53/Skydiffusion) | README 链接到 opendatalab 主仓库；提到 CVUSA 下载脚本，但该仓库本身未包含脚本或权重文件 | 不能仅凭 README 中的命令宣称 CVUSA/G2A-3 权重可用 |
| [另一公开代码仓库](https://github.com/SkyDiffusion0307/skydiffusion0307.github.io) | main 树有 586 条目，包含旧版投影、模型和评估代码；README 没有新的 checkpoint 下载链接 | 尚未核验为作者正式实验 release；旧版 CVACT 默认参数不能升级为 Disaster 标定 |
| 官方项目 `gh-pages` 分支 | PDF 文件主要是数据/结果展示图，没有找到补充协议或正式划分清单 | 展示图不能代替实际实验配置或 split manifest |
| Hugging Face 作者 / 权重上传者模型列表 | `Yejy53` 的公开列表是 BEV 检索与其他生成项目；`SkyDiff1109` 列表仅见现有 SkyDiffusion checkpoint 仓库 | 未发现可核验的 G2A-3 三任务专用 diffusion 权重 |
| [官方数据仓库](https://huggingface.co/datasets/Yejy53/G2A-3/tree/main) | 根目录是三个 ZIP、24 字节 README 和 `.gitattributes`；结合前轮 Disaster/History ZIP 核验，未找到正式名单 | 按地域 4:1 的具体 IDs/边界仍需作者提供；UAV 官方嵌套内容未完成全量核验 |
| [旧 README 链接的数据仓库](https://huggingface.co/datasets/SkyDiff1109/G2A-3/tree/main) | 仍可获取，根目录也是三个 ZIP 和简短 README；Disaster/History 发布的容器 hash 与 `SkyDiff/G2A-3` 对应文件元数据相同，UAV 又是不同版本 | 说明旧版数据也曾有官方链接背景，但不能据此选定 ICCV 正式实验使用的 revision 或补齐划分 |
| [数据仓库讨论](https://huggingface.co/datasets/Yejy53/G2A-3/discussions) | 通过镜像 API 查询到 1 个添加任务类别/论文链接的 PR | 没有找到相应材料发布讨论 |
| [权重仓库讨论](https://huggingface.co/SkyDiff1109/SkyDiffusion_ckpt/discussions) | 镜像 API 返回空讨论列表 | 没有找到补充权重或评估协议 |

原 README 中 `SkyDiffusion/SkyDiffusion-code` 和旧项目仓库 `longhz140516/SkyDiffusion.github.io` 的公开 GitHub API 查询返回 404。404 不能区分删除、重命名或私有化，不据此声称材料从未存在。

Hugging Face 主站讨论 API 连接超时，后通过 `hf-mirror.com` 获取公开讨论列表；镜像快照不保证与主站完全实时同步。完整成功/失败状态、来源 URL、时间和文件哈希见 `results/author_materials_search/source_index.json`。未把网络错误解释为“没有材料”。

## 已确认的联系渠道

- 作者：Junyan Ye（GitHub `yejy53`）。
- 公开科研邮箱：`yejy53@mail2.sysu.edu.cn`。
- 来源：[作者主页](https://yejy53.github.io/) 的 `mailto:` 链接，并与其主页源码 `_config.yml` 相互核验。
- 本地证据：`results/author_materials_search/author_home.html`、`author_config.yml`。
- 官方 GitHub 讨论入口：[opendatalab/skydiffusion Issues](https://github.com/opendatalab/skydiffusion/issues)。

此邮箱是作者公开发布的联系地址，不表示本轮已验证其送达性或取得联系。通讯作者主页配置中出现 `you@example.com` 模板字段，未将其当作真实邮箱。

## 最小材料请求

为了先完成 Disaster 对齐，不必一开始索要所有训练数据和全部模型。建议优先请求一个可以重现 ICCV 表 3 Disaster 行的最小实验包：

1. 对应 checkpoint 的下载 URL、文件 hash、训练配置及准确代码 commit；若不能公开权重，至少提供正式测试预测图。
2. Disaster 正式 train/test sample IDs 与地域边界、所用数据仓库 revision；验证 split 是否避免地域重叠。
3. 预处理/投影配置：米每像素、相机高度符号、单视图曲率 λ、全景重采样尺寸、北向和垂直坐标约定。
4. 可运行评估命令与依赖版本：原始 target 或 VAE reconstruction、SSIM 通道/窗口、LPIPS 网络/版本、PNG 量化、FID 权重和参考统计集。

英文邮件及 GitHub Issue 草稿已放在 `SkyDiffusion作者材料请求草稿.md`，**尚未发送**。草稿不包含机器路径、个人身份声明或内部 checkpoint 路径；如联系作者，由用户补充身份和发送地址。

## 后续处理边界

- 公开链接一旦获得即可按来源、版本、hash 下载核验；不能绕过权限获取私有文件。
- 邮件或 Issue 是对外联系，需要用户确认内容和使用的账号；本轮只准备草稿，不冒用身份发送。
- 作者未回复前不把默认参数、随机划分或独立域内微调标记为作者正式实验。
- 收到材料后，先补齐证据和锁定配置，再跑完整正式测试；当前 benchmark 缺口不会仅因为发出请求而消失。
