# 本地使用说明

当前版本为 `0.1.0`，面向已有 VASP 和 Slurm 使用权限的单个研究人员，支持非磁性 PBE 的有序周期结构。无需 LLM，也不连接旧版 AI_AGNET 的数据库。[真实验收记录](../validation/results/campaign.json)持续记录各任务状态、求解器版本、收敛检查和文件哈希，完成情况以记录中的状态字段为准。

## 1. 安装

先获取源码并进入项目目录；如仓库访问需要认证，请先完成 GitHub 登录：

```bash
git clone https://github.com/cuiqirui99/vasp-slurm-agent.git
cd vasp-slurm-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
vasp-agent ui
```

需要 macOS 或 Linux、Python 3.11 或以上和本机 SSH 客户端；当前不支持原生 Windows。界面只监听 `127.0.0.1`。不要将其直接转发成公网服务；它没有面向多用户的登录和租户隔离功能。

远端登录与计算环境需要 Python 3.8 或以上、Bash 和 `sha256sum`，Slurm 需提供 `sbatch`、`squeue`、`sacct`、`scancel`。需要加载模块时，将对应命令填入集群配置。

源码的 `examples/` 提供小型 CIF 输入，安装后的 wheel 也包含这些示例。没有源码目录时，可提取 Si 结构再上传：

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
Path("Si.cif").write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
PY
```

这些是起始结构，不是已收敛的参考结果。

## 2. 配置集群

先确认在终端可以通过 SSH 登录目标集群，并核对首次连接的主机指纹。在「集群配置」页设置：

| 配置 | 含义 |
|---|---|
| SSH 主机、用户、端口 | 与你已验证的 SSH 连接一致 |
| 远端任务根目录 | 有写权限、空间足够的绝对路径 |
| VASP 启动命令 | 集群认可的命令，例如管理员要求的 `srun vasp_std` |
| POTCAR 根目录和元素映射 | 你有权使用的赝势目录；按研究方法选择势，不自动认定默认势正确 |
| partition、account | 有权使用的 Slurm 队列和计费账户 |
| MPI task 数、walltime | 每个阶段请求的资源与时间上限 |
| 环境命令 | 模块加载等命令，每行一条，只使用自己了解的配置 |

配置保存在本机 JSON 文件中，默认 `~/.config/vasp-slurm-agent/cluster.json`；任务目录默认 `~/vasp-slurm-agent-runs`。页面侧栏可以修改两者。每次生成任务时会保存配置快照，之后修改全局配置不会自动修改旧任务。

优先使用 SSH 密钥或 SSH agent。只有密码登录的站点，可以在侧栏的密码框中输入：密码只留在当前会话内存，显式检查、提交、恢复或取消时才交给对应进程，不写入 JSON。可用「清除会话密码」移除它；已经启动的后台 worker 会保留登录所需的进程环境直至退出。CLI 可用 `vasp-agent watch <任务目录> --password` 或 `vasp-agent doctor --password` 私密输入，也支持环境变量 `DFT_AGENT_SSH_PASSWORD`。不要把密码写进命令历史或上传文件。依赖交互式 MFA 的集群需要按站点要求建立连接，本版本不宣称已适配所有 MFA 流程。

保存配置后可点击「检查 SSH / Slurm / VASP 环境」。只有明确点击才发起远端检查，不提交计算作业。

## 3. 准备并确认

上传 CIF 或 POSCAR，检查化学式、原子数、晶胞和坐标，然后选择：

| 任务 | 目标 |
|---|---|
| `relax` | 优化结构，可选择同时优化晶胞 |
| `scf` | 对输入结构计算自洽电子态 |
| `bands` | 同一输入结构的 SCF → 能带；提交前检查路径 |
| `dos` | 同一输入结构的 SCF → 态密度；提交前检查网格 |

四条路径目前使用非磁性 PBE（`ISPIN=1`）。`bands` 和 `dos` 不会自动先优化结构；需要优化时先完成 `relax`，检查结果后再使用其结构。只接受元素明确、占位完整的周期结构。

设置 ENCUT、Γ 中心 k 网格、EDIFF、ISMEAR、SIGMA；优化任务还需要力阈值、步数上限和是否优化晶胞。页面上的力阈值为正数，其对应 INCAR 使用负的 EDIFFG。默认数值只用于开始配置，不能替代收敛性测试。

点击「生成输入并预览」只在本机准备任务。核对阶段、完整参数、账户和资源后，勾选确认并点击「确认并提交」，才会启动后台 worker 和远程提交。

## 4. 跟踪、恢复和取消

任务页每 5 秒刷新保存的状态和 Slurm job ID。`queued` 表示排队，`running` 表示计算，`collecting` 表示收集；`succeeded` 表示流程判据通过，具体物性仍需核查。`needs_attention` 或 `failed` 时查看阶段结果和错误记录，不要将其算作成功。

关闭浏览器不会主动取消远端任务。本机关闭或重启后，Slurm 作业可能仍在运行；重新打开界面，在「运行记录」选择原目录，再点「恢复后台监控」。恢复应沿用保存的 job ID；不要仅因页面断开就重新生成同一任务。

远端操作连续失败三次会暂停监控，显示 `needs_attention`。检查错误后，点击「重新连接 / 回收」重新关联原作业并收集结果；它不会更改科学参数，原来未收敛的计算不会因此变成成功。

「取消此任务」针对当前任务，不应当作取消已确认，直到保存状态和集群队列都反映取消结果。需要检查时，用站点提供的 `squeue` / `sacct` 命令核实。

「打包当前结果」生成当前快照，再点击「下载 ZIP」。分享前检查计算输入、账户/路径等元数据；VASP 可执行文件、POTCAR 和凭证不得作为公共交付内容。

通过检查的阶段会显示最终总能量、最大原子力和最终结构，并提供单独的 CIF 下载。未通过检查时显示拒绝原因，不把已有结构文件标成成功结果；原始输出仍可通过 ZIP 检查。

## 5. 命令行

可用命令包括 `init`、`doctor`、`prepare`、`watch`、`resume`、`status`、`cancel`、`bundle`。用以下方式查看实际参数：

```bash
vasp-agent --help
vasp-agent prepare --help
vasp-agent watch --help
```

`watch` 可以提交一个已经准备好的任务，运行前须先核对输入。`resume` 用于恢复暂停任务并继续监测。只读查看请使用 `status`。

完整命令示例见 [Si 结构优化、Al SCF 和 MgO DOS 教程](examples.md)，包括提交前检查、恢复和结果文件位置。

## 使用边界

磁性、DFT+U、SOC、杂化泛函、声子、NEB、缺陷等需求不在这四条方法范围内。本工具不会根据失败自行调整科学参数。数值精度和具体材料适用性需要按研究目标另行检查。

开发验收依据[冻结方案](../validation/protocol.v1.json)，真实运行记录保存在 [`validation/results/`](../validation/results/)。代码与文档采用 [MIT 许可证](../LICENSE)，外部求解器条款见 [NOTICE.md](../NOTICE.md)，引用信息见 [CITATION.cff](../CITATION.cff)。
