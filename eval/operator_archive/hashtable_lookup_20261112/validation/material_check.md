# material_checked 记录（下载侧，NDF §6.1 / §5.1）

- 时间：2026-11-12；执行人：lvhao（归档人，**非独立复核**）
- 归档 revision：3；固定 commit `4359f297bdc855b91870023df2605fc7a31ff916`
- Release：https://github.com/PTO-ISA/SuperNPUBench/releases/tag/hashlk-20261112-r3
- 工具：`gh` 2.96.0、Python 3.9.6、macOS arm64
- 下载目录：`/tmp/ndf_material_check2/downloads`（与本机仓库无关的新目录）

## 1. 获取与哈希

```text
gh release download hashlk-20261112-r3 -R PTO-ISA/SuperNPUBench -D downloads   # rc=0
shasum -a 256 -c assets.sha256                                                 # rc=0，11/11 OK
```

- 12 个附件（11 个资产 + `assets.tsv`），合计 261,880 KiB。
- 字节数与 `assets.tsv` 的 `bytes` 列逐个一致；SHA256 11/11 匹配。
- ELF 身份：`A-REF-ELF` `960ba161…`、`A-R1-ELF` `2324ea4c…`、`A-R2-ELF`
  `35218ed0…`、`A-R3-ELF` `38a5f34a…`，与 `results/*/manifest.json` 一致。

## 2. 归档包内部清单

```text
cd extracted/hashlk_archive_r3/archive/hashtable_lookup_20261112   # A-ARCHIVE-R3 快照路径；
# revision 5 起仓库路径为 eval/operator_archive/hashtable_lookup_20261112
sha256sum -c FILE_MANIFEST.sha256    # rc=0，56/56 OK（55 文件 + JSON 自身排除）
python3 -m json.tool FILE_MANIFEST.json  # 可解析
```

解压后共 248 个文件（归档 + 4 组 trace + PipeView + API 快照）。

## 3. 路径与内容扫描

使用归档内的 `build/scan_paths.py`（sha256 `c34f2db2f47f513a95ef6c1f86294c49b5146db7a6bdf0406900d5bccec2abd5`）：

| 范围 | 文件数 | 命中 | 个人路径（forbidden） | 结果 |
|---|---:|---:|---:|---|
| 下载的 12 个附件 | 12 | 109 | **0** | rc=0 |
| 解压后的 payload | 248 | 109 | **0** | rc=0 |

- 109 条命中全部是 `/tmp` 临时路径（历史日志 argv、负控示例、验证记录中的命令），
  逐项判定见 `PATH_SCAN.md`；无 `/Users/<name>`、`/home/<name>`、`C:\Users\<name>`
  或用户名命中。
- 压缩包按“先解压再扫描”处理；trace JSON 流式解压后扫描。
- 首次上传的 6 个 tar.gz 在 tar header 中带有本机用户名；已用
  `--uid 0 --gid 0 --uname ndf --gname ndf` 重新打包并 `--clobber` 上传，
  新 hash 已写入 `ASSETS.md` 与 `assets.tsv`。此前的包未再被引用。

## 4. 结论与边界

- **material_checked（下载侧）通过**：资产可获取、字节数/SHA256 一致、归档内部
  清单自校验通过、个人路径 0 命中、命令与归档声明一致。
- 未执行：实际运行复现（`run_reproduced`）与独立目录重建（`build_reproduced`）；
  二者必须由非归档人完成，尚未发生。
- 本记录不能升级为 `run_reproduced`，也不能替代数值验证。

## 5. revision 5 路径迁移后的归档包

- `A-ARCHIVE-R5`（`hashlk_archive_r5.tar.gz`，220,886 B，sha256
  `470923ca70b30ea6488c858b475f472c94fdc9b9fbc50c2b41bb357100d82441`）：
  从 Release 下载后哈希一致；解压内容与
  `eval/operator_archive/hashtable_lookup_20261112/` 工作树逐文件一致
  （`diff -r` rc=0）；扫描 61 个文件、个人路径 0 命中。
- `A-ARCHIVE-R3` 保留为 revision 3 的历史快照（旧路径 `archive/...`），不覆盖。
