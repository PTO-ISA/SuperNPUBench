# 依赖：PTO API 快照（PR303 期）

| 项 | 值 |
|---|---|
| 位置 | `outputs/hash256_soa_20261105/optimization/dependencies/tileop-api`（169 文件） |
| 清单 | `MANIFEST.sha256`（逐文件 sha256，共 169 行） |
| 用途 | 编译宽 lane 元素区域（`TPARTELEMENT` 支持 256 元 tile） |

## 缺口

- `GAP-API-1`：**来源 commit/PR 未确认**。它既不是主 `linx-toolchain-build` 资源目录里的
  `tileop-api`（那个 `TPARTELEMENT` 只接受 32 元区域），也没有记录对应的上游 PR。
  因此该依赖目前**无法从上游文档重建**；恢复只能靠这份本地快照。
- 建议：确认上游 PR/commit 后固定为 `dependency_id`，或把目录发布为依赖包。
