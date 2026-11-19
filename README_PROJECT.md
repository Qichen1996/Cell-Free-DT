# Cell-Free 基站节能与 QoS 策略研究

本项目研究 Cell-Free 网络中基站功耗与不同 UE 的服务质量之间的权衡，包含原始统一 QoS 设计（ORI）、区分 UE 类型的 PPO 策略、全局 critic/reward 的 MAPPO 对照，以及使用离线轨迹训练的 Decision Transformer（DT）和带偏好 prompt 的 PDT。

本文是整个研究工作区的导航文档，覆盖本地与 Vast 两端。配置、模型身份和结果以对应实验的 manifest、实际配置、模型 SHA256 和独立核验记录为准；目录名称与旧 README 中的运行状态不能单独作为依据。

文档整理日期：2026-11-03。运行中的任务状态会变化，本文不充当实时任务面板。

## 1. 项目位置

| 内容 | 本地 | Vast |
| --- | --- | --- |
| 本研究工作区：调度、核验、绘图、汇总 | `C:/Users/Admin/Documents/Codex/2026-10-05/new-chat` | — |
| NEW：UE differentiation、PPO、DT/PDT | `E:/PycharmProjects/Cell-Free-DT-new` | `/workspace/Cell-Free-DT-new` |
| ORI：原始统一 QoS 设计 | `E:/PycharmProjects/Cell-Free-DT-ori` | `/workspace/Cell-Free-DT-ori` |
| 独立核验与数据索引 | 本工作区 `pc_next_vast/` | 各实验目录中的 manifest、verified、complete 文件 |
| 论文/PPT 图及导出文件 | 本工作区 `outputs/` | 原始 CSV 在相应项目 `analysis/sim_stats/` |

本地与云端不是自动同步镜像。找到同名文件后仍需核对哈希；本地有汇总 JSON 不等于已备份对应模型或完整轨迹。

## 2. 代码结构

NEW 与 ORI 共有的主要入口：

```text
agents/             策略网络、动作分布及 agent 实现
trainers/           强化学习优化与训练逻辑
env/                多智能体环境与接口
network/            基站、UE、信道、网络、功耗与状态更新
traffic/            业务流量模型
data/               环境所需数据
arguments.py        命令行参数
config.py           项目配置
train.py            强化学习训练入口
simulate.py         策略仿真入口
results/            强化学习模型与训练输出
analysis/           CSV、轨迹、分析脚本与 notebook
experiments/        带版本的实验 runner、计划、快照、日志和完成记录
```

NEW 另外包含：

```text
decision_transformer/   DT/PDT 模型与训练相关实现
train_dt.py             离线序列模型训练入口
test_dt.py              DT/PDT 推理与仿真入口
trajectory_io.py        云端轨迹压缩读写支持
dt_results/             各数据版本下的 DT/PDT 模型
```

工作区中的 `install_*.py`、`launch_*.py`、`resume_*.py` 是具体实验的编排脚本，`audit_*.py` 负责核验，`plot_*.py` 负责绘图。这些脚本经常绑定特定路径和模型，并非通用命令行工具；执行前先看对应计划，避免重复启动或覆盖结果。

## 3. NEW、ORI 和 MAPPO 的区别

| 分支/方法 | 主要作用 | 注意事项 |
| --- | --- | --- |
| ORI PPO | 原始统一 QoS 设计，对照 UE differentiation | 常用模型 actor 输入为 98 维；使用 ORI 环境与原始奖励定义 |
| NEW PPO/IPPO 实验 | 区分 Stringent、Sensitive、Tolerant，扫描训练 reward weights | 常用 actor 输入为 110 维；以实际 critic 输入和 reward 定义识别算法 |
| 本地 MAPPO benchmark | actor 使用局部观测，critic 使用共享全局状态，所有 agent 使用全局团队奖励 | 通过适配器显式修改环境，不能仅凭 `use_centralized_V=true` 宣称已完成算法转换 |
| DT | 用离线轨迹学习由 RTG 条件控制的策略 | 没有 PC 偏好 prompt |
| PDT | 在 DT 输入上加入偏好 prompt | 当前主线 prompt 为 `PC weight / 3.3` |

ORI 的 98 维模型与 NEW 的 110 维模型不能互换。模型加载成功也不足以证明环境正确，仍需检查布局、邻居、cluster、traffic 和动作定义。

本地 MAPPO 适配实现为本工作区 `mappo_global_critic10.py`，使用 10 维聚合全局状态，并广播相同的团队 reward；它是本项目对照实现，不应写成与其他仓库逐项完全相同的 MAPPO。

## 4. 七个原始环境

以下为当前七场景离线数据计划中的部署参数。历史实验存在其他 cluster/布局版本，复现时以该次实际配置为准。

| 标签 | 流量 profile | 区域边长（m） | AP 数 | AP 间距（m） | cluster |
| --- | --- | ---: | ---: | ---: | ---: |
| A | A / traffic 0 | 1000 | 16 | 250 | 0.98 |
| B | B / traffic 1 | 1000 | 16 | 250 | 0.90 |
| C | C / traffic 2 | 1000 | 16 | 250 | 0.90 |
| B4 | B | 1000 | 4 | 500 | 0.90 |
| B36 | B | 1000 | 36 | 166.67 | 0.90 |
| B500 | B | 500 | 16 | 125 | 0.90 |
| B2000 | B | 2000 | 16 | 500 | 0.95 |

标准长期评估为七天、10080 个记录步。B 主线常用邻居距离 0.25 km、最少连接数 1；其他部署不应直接套用 B 的配置。部分输出仍在 `.../B/B/SEEDn` 下，因此不能仅凭中间目录 `B` 判定区域大小或 AP 数。

## 5. 权重、seed 和 checkpoint

NEW 四个 reward weights 按名称记录为：

```text
w_str / w_sen / w_tol / w_pc
```

常用统一采集/评估权重为 `40 / 0.5 / 0.5 / 3`。B 场景 PPO trade-off 训练通常固定前三项 `80 / 0.2 / 0.2`，改变 PC weight。训练权重与离线轨迹 reward 的重算/记录权重必须分开保存。

ORI 使用 `w_qos / w_pc`，另有 ExtraQoS 参数，其含义不应直接映射为 NEW 三类 QoS 权重。原始 B 对照为 QoS50、PC1.2、ExtraQoS0；后续有 QoS40、QoS45、PC1.2 的独立重新训练。

三种 seed 是独立概念：

- **training seed**：决定策略训练随机性；每个 seed 生成不同模型。
- **evaluation seed**：决定评估环境；通常每个模型评估 seed1–5，扩展实验使用到 20 或 40。
- **collection seed**：决定离线轨迹采样环境；当前 selected-single 数据已核验为 seed1–20。

历史 seed101–105 与 seed1–5 不能直接合并成相同测试集。`CP200`、`CP230` 等表示保存检查点；`final` 表示训练结束权重，不能在没有元数据的情况下等同于最近的编号 CP。

## 6. PPO trade-off 与轨迹来源

当前单模型选点版本索引为本工作区：

```text
pc_next_vast/PPO_last21_pc27_seed1_cp220.json
outputs/B_tradeoff_latest_20261030/PPO_last21_pc27_seed1_cp220.png
```

| PC weight | training seed | checkpoint |
| --- | ---: | --- |
| 0.6 | 1 | 240 |
| 0.9 | 1 | 230 |
| 1.2 | 3 | final |
| 1.5 | 3 | 240 |
| 1.8 | 5 | final |
| 2.1 | 5 | final |
| 2.4 | 3 | 240 |
| 2.7 | 1 | 220 |
| 3.0 | 1 | 200 |
| 3.3 | 1 | 200 |

这些点来自查看评估结果后的探索性选点，不能当作统一 checkpoint、统一选择规则下的无偏算法排名。论文应交代选点过程；多训练 seed 的均值结果须与此单模型曲线区分。

这 10 个模型已各采集 20 条轨迹，共 200 条，云端目录：

```text
analysis/trajectory/tradeoff_selected_single_seed1to20_v1/{model_tag}/SEED{1..20}.pkl.zst
experiments/tradeoff_selected_single_collect200_v1/dataset-manifest.json
```

本地独立核验：`pc_next_vast/tradeoff_selected_single_collect200_verified.json`。增加到每模型 40 条是后续授权目标；不能把授权目标视为已经完成的 400 条数据。

## 7. DT/PDT 数据与模型版本

模型路径均相对于 NEW 项目 `dt_results/`。

| 版本 | 数据特点 | 用途 |
| --- | --- | --- |
| `selected_seven_pc33_v1/PDT` | 原七环境 PPO 数据，B 包含多 PC 偏好 | 常称“老 PDT”，作为 prompt 基线 |
| `selected_seven_baseline_aug_v1/PDT` | 在原数据基础上增加 Always_on、Auto_SM1 | baseline 数据增强对照 |
| `selected_seven_Brefresh150_v1/{PDT,DT}` | 旧 B 85 条替换为 B150，其他六场景及 baseline70 保留；517 episodes / 7012 AP trajectories | refresh150 对照 |
| `selected_seven_Bsingle200_v1/{PDT,DT}` | 旧 B 全部替换为 selected-single B200；其他六场景保留，不包含 Always_on/Auto_SM1；497 episodes / 6612 AP trajectories | 当前 single200 主线 |

`single200` 指 B 数据由 10 个各自选定的单模型提供，总计 200 条，并非只训练 200 步或只有一个模型。

最近两版的训练设置：seed1，从头训练；1800 轮 × 每轮150更新 = 270000更新；batch256、context8、3层、hidden128、1 attention head、dropout0.1；AdamW，初始 LR5e-4、weight decay1e-4，沿用原 LambdaLR；RTG gamma1、scale1。这些是具体版本的配置，不代表所有历史模型都相同。

PDT prompt 为 `PC weight / 3.3`，例如 1.8 对应约0.54545。prompt 扫描通常使用 PC weights 0.6、0.9、…、3.3。DT 不输入该 prompt。

已按用户要求删除云端三个早期 PDT 模型及其专属仿真：`B1000_preference_v1/PDT`、`B1000_preference_v2/PDT`、`PDT_return_prompt_seven_ctx8`。同名共享轨迹和 DT 未随之删除。删除记录为 NEW 项目 `experiments/early_PDT_deletion_record.json`。

另一个旧数据集 `tradeoff_selected123_v1` 的 seed11–15 轨迹也已按授权删除；其 manifest/汇总属于历史记录，不能当作现存可训练文件。

## 8. 轨迹、RTG 与模型输入

B 标准单条七天轨迹的主要数组形状：

```text
observations : (10080, 16, 110)
actions      : (10080, 16, 2)
rewards      : (10080, 16, 1)
```

其他环境 AP 数不同，第二维随之变化。一个网络 episode 可拆为多个 AP 训练序列，因此 episode 数不等于 AP trajectory 数。

对 AP i，未折扣 RTG 定义为：

```text
RTG[t,i] = sum(reward[k,i] for k=t..T-1)    # gamma=1
```

汇报单条轨迹的初始 RTG 时，通常先对每个 AP 累计七天 reward，再在 AP 间平均。PPO 执行策略不输入 RTG；DT/PDT 才需要目标 RTG。

当前固定目标 RTG 对照常用 A=-288.18793981031047、B=-426.00145441593037、C=-849.5081207960611。历史七场景模型使用过不同目标，必须检查每次仿真的实际参数。

“与 PPO 匹配 RTG”实验使用相同 seed 的 PPO 轨迹初始 AP 平均 RTG，广播给 PDT 的 AP 输入。这是利用参考轨迹结果的诊断对照，不等于在线部署时能预先获得该 RTG，也不等于给每个 AP 分别输入其原始 RTG。

轨迹读取应使用项目的压缩读写工具并核验 manifest。PKL 只读取受信任的本项目文件。

## 9. 统一评价口径

### 9.1 功耗

使用 `reward.csv` 中 `pc_kw` 的时间均值。等长 seed 的均值再平均，与合并全部时间步后平均相同。

```python
pc_kw = reward['pc_kw'].mean()
pc_w = 1000 * pc_kw
```

该字段已是网络功耗，不能再次除以 AP 数。不要用 `net_stats.csv` 的 `avg_pc` 替代。图中可写 **Average PC over One Week (kW)**。

### 9.2 三类 packet drop ratio

按 `ue_stats.csv` 的 delay budget 分类：Stringent=0.05、Sensitive=0.15、Tolerant=0.3。浮点分类应使用容差比较。

对同一模型、同一场景、同一类别，先跨评估 seed 累加 packet 计数，再计算：

```text
drop (%) = 100 × sum(dropped_packets)
                 / [sum(dropped_packets) + sum(completed_packets)]
```

这是 packet-pooled ratio，不是先算每个 seed 的百分比再取平均，也不是平均瞬时 drop。若总包数为零，应标记未定义而非擅自记零。

常用目标为三类 pooled drop 均不超过0.5%；这不保证每个 seed 均达标。只绘制 STR 时仍需明确说明省略了另外两类。

### 9.3 reward 和 sleep mode

不同训练 weights 下的原始 reward/RTG 不能直接按高低比较策略优劣。算法对比优先报告相同评估口径下的 PC 和三类 drop。

Sleep mode 图统计真实状态 Active/SM1/SM2/SM3。日内曲线可按多个 seed × 七天中的相同小时平均 AP 数；四类之和应等于 AP 总数。若与 PC 叠图，PC 使用独立右轴，写清单位与聚合窗口。

## 10. 已有结果的主要索引

以下均位于本工作区 `pc_next_vast/`，原始 CSV 路径通常保存在 JSON 中。

| 文件 | 内容 |
| --- | --- |
| `PDT_original_prompt_AC_verified.json` | 老 PDT 在 A/C 的 prompt 扫描 |
| `PDT_Brefresh150_seven_prompt_verified.json` | refresh150 七场景及 prompt 结果 |
| `PDT_Bsingle200_prompt_ABC_verified.json` | single200 在 A/B/C 的 prompt 结果 |
| `PDT_two_versions_A_seed20_verified.json` | 两版 PDT 在 A 的20seed对照 |
| `PDT_seedmatched_RTG_complete_comparison.json` | 老 PDT 的 PPO 匹配 RTG 对照 |
| `tradeoff_selected_single_collect200_verified.json` | 当前 B200 轨迹独立核验 |
| `tradeoff_selected123_collect_cross_verified.json` | 历史采集与三个2.1模型跨七场景结果 |
| `ORI_B_cp160_180_190_verified.json` | 原始 ORI 三个 CP 与 final 参考 |
| `ORI_B_qos40_pc12_seed1_ep250_verified.json` | ORI QoS40 训练核验 |
| `ORI_B_qos40_ep250_eval_verified.json` | ORI QoS40 final 仿真核验 |
| `B_all_fixed_final_models_verified.json` | B PPO final 模型补测核验 |

主要绘图目录：

```text
outputs/B_tradeoff_latest_20261030/       PPO reward weight trade-off
outputs/PDT_refresh150_vs_single200/     两版 PDT 的 ABC prompt 对照
outputs/PDT_A_seed20/                    A 场景5seed与20seed对照
outputs/PDT_original_prompt_AC/          老 PDT 的 A/C prompt 曲线
outputs/PDT_seedmatched_RTG_complete/    RTG 匹配实验
outputs/ORI_vs_tradeoff/                 ORI 与 UE differentiation 的对照
outputs/ori_new_figures/                 sleep mode 与 PC 叠图
outputs/cross_scenario_scatter/          跨场景 PC/drop 散点图
```

A/C 已出现在七场景 PDT 的训练数据中，因此这些测试不能笼统称为“未见场景 zero-shot”。若 A/C 没有对应偏好标注，可讨论未见过的场景与偏好组合，但需具体描述训练覆盖。

## 11. ORI 实验记录

ORI 训练和评估全部在 ORI 项目下进行，不使用 NEW 模型与仿真环境。

| B 训练版本 | QoS / PC / ExtraQoS | 训练预算 |
| --- | --- | --- |
| 原始 `mx5fmmsf` | 50 / 1.2 / 0 | 200轮，2016000步 |
| `ORI_B_qos40_pc12_seed1_ep250` | 40 / 1.2 / 0 | 250轮，2520000步 |
| `ORI_B_qos45_pc12_seed1_ep250` | 45 / 1.2 / 0 | 250轮，2520000步 |

后两版 seed1、21 rollout workers ×480，每轮10080样本，5训练线程，actor/critic LR5e-4固定、不使用 LR 衰减。完成训练不等于完成仿真。

QoS40 的 CP180/200/220/230/240 已有五 seed 仿真；final250 也有单独核验。原始 QoS50 final 参考约329.7149W，STR/SEN/TOL drop 为0.347826% / 0.000289% / 0%。这些数字仅对应所列历史配置与评估集。

ORI 的确定性动作重放已与原始 CSV 前段功耗核对；不能用随机动作仿真与确定性旧结果直接比较。部分历史 runner 在仿真后仅汇总阶段失败，应看独立核验结果，避免重复仿真。

## 12. 本地 MAPPO 对照

```text
local_MAPPO_B_global_40_1_1_pc05/
local_MAPPO_B_global_50_2_2_pc03/
train_local_MAPPO_B_global.py
train_local_MAPPO_B_global_50_2_2_pc03.py
eval_local_MAPPO_B_global.py
pc_next_vast/mappo_comparison/comparison.md
```

四项分别为 STR/SEN/TOL/PC weights。两组训练使用不同目标；评估前必须确认共享全局 reward 和 critic 适配已生效，不能只查看训练算法名。第一组有 final 五 seed 核验，第二组训练完成与否、是否新增评估，应检查自己的 complete 和 evaluation 目录。

## 13. 安装与复现

项目的 `requirements.txt` 是较早的依赖列表，含旧版 NumPy、pandas、Gym、W&B；PyTorch 版本曾被注释。不要把它视为已验证的现代环境 lockfile。建议在独立环境中保留实际 Python、PyTorch、CUDA、NumPy 版本与安装清单。Vast 已有环境位于 `/venv/main`，训练使用 `/venv/main/bin/python`，不是系统 Python。

复现顺序：

1. 确认 NEW/ORI 分支、源代码版本、依赖环境。
2. 从实验 `plan.json`、`specs.json`、`actual_config.json` 读取实际参数与模型来源。
3. 核对 actor/critic 或 model.pt 的 SHA256、输入维度、动作空间。
4. 核对 traffic、区域、AP布局、邻居、cluster 和 reward weights。
5. 完成小规模前向/环境预检后，使用独立新实验名启动。
6. 完成后独立核验 CSV 或轨迹，再聚合和绘图。

可先查看入口参数，不会启动训练：

```bash
cd /workspace/Cell-Free-DT-new
source /venv/main/bin/activate
python train.py --help
python simulate.py --help
python train_dt.py --help
python test_dt.py --help
```

ORI 使用对应目录中的 `train.py`、`simulate.py`。复杂历史实验优先依据其保存的 runner 与计划复现，本文不提供省略关键环境覆盖项的“一键通用训练”命令。

## 14. 完成、核验和存储

云端长期任务由 supervisor 管理。每次操作前完整读取 `/etc/vast-agents-guide.md`。只读查看示例：

```bash
supervisorctl status
df -h /workspace
tail -n 30 /tmp/<experiment>.log
```

`EXITED` 不一定失败：一次性 runner 正常退出后也会是 EXITED。完成判断需要日志完成标记、预算计数、complete.json 和输出哈希共同支持。不要仅凭进程消失重启任务。

典型实验目录：

```text
experiments/<experiment>/
  runner.py / plan.json / specs.json
  models/                 被评估权重快照与 provenance
  logs/ / state.json
  dataset-manifest.json   如果包含轨迹采集
  summary.json / complete.json
```

仿真独立核验：检查约定的五个 CSV 非空、reward 10080行、PC有限、verified 中 seed/环境/模型哈希匹配。不同 runner 的五个 CSV 名称以其验证器为准。

轨迹独立核验：逐文件检查压缩包可读、SHA256、数组长度/维度/有限性、collection seed、training seed、CP、reward weights 和模型 provenance。

云端 `/workspace` 不代表已永久备份。stop/start 与 recycle/destroy 的持久性不同；删除云端结果前应确认本地对应原始文件和哈希，而非仅有图片或汇总表。所有清理均需限定具体实验范围，保留共享依赖和删除记录。

## 15. 后续维护约定

- 新模型、新数据集、新 RTG 或 prompt 实验使用独立版本名。
- 训练、轨迹采集、仿真与绘图的参数分别记录，禁止以修改说明文件代替实际重跑。
- 相同图内使用匹配的评估 seed、仿真长度和聚合方法。
- 历史 README、RTG_README 和调度脚本可能保留旧状态；优先使用实际输出与最新独立核验。
- 本 README 不会自动启动、停止、训练、仿真或删除任何实验。

