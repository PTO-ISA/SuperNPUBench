# SuperNPUBench Plan Board

静态工作看板，用于维护和展示：

- 已经发布的特性；
- 当前工作状态与下一步；
- 版本发布计划、目标日期和每个版本包含的特性；
- 数值验证、功能回归和性能分析结果；
- 算子、编译器、TileOP 与模型的版本基线。

## 交互式维护

打开页面后点击右上角“管理看板”，可以通过表单维护全部内容：

- 修改看板标题、负责人、更新时间和当前重点；
- 新增、修改、删除和排序摘要指标、发布特性、当前工作、计划、验证结果和版本基线；
- 导入或导出完整看板数据；
- 恢复仓库中保存的默认版本。

当前工作卡片支持原地编辑：

- 点击“＋ 添加工作卡”即可快速创建任务，不需要先打开管理窗口；
- 也可以点击工作卡列表末尾的“＋”卡片快速创建，拖动卡片左上角手柄可调整顺序，排序结果会自动保存；
- 任务 ID、领域、标题、工作说明、当前问题总结和下一步均可直接在卡片上点击编辑；
- 状态和优先级可直接选择，ISA、算子、编译器和 gfrun 四个阶段在同一行紧凑展示，并支持新增、重命名和删除阶段；
- 修改后卡片会标记为未保存，点击卡片底部醒目的“保存修改”即可持久化，也可以按 `Ctrl/Command + Enter` 快速保存当前卡片。

计划、验证结果和发布特性卡片右上角也有“编辑”按钮，点击后会直接打开并定位到该条记录，不需要再从内容管理器中查找。

页面展示顺序以日常工作为主：当前工作、版本发布计划、验证结果、版本基线，已发布特性作为历史归档放在最后。

点击“保存全部更改”后，修改保存在当前浏览器的 `localStorage`，刷新页面不会丢失。浏览器本地修改不需要登录。

要让所有访问 GitHub Pages 的人看到更新：

1. 在管理窗口点击“导出数据”；
2. 用下载的文件替换 `benchmark/plan_board/plan_data.json`；
3. 提交并推送到 `main`，Pages 工作流会自动更新网站。

也可以直接编辑 [`plan_data.json`](./plan_data.json) 后提交。

## 本地预览

在仓库根目录执行：

```bash
python3 -m http.server 8077 --directory benchmark/plan_board
```

然后访问 `http://127.0.0.1:8077/`。由于页面通过 `fetch()` 加载 JSON，不建议直接双击 `index.html`。

## GitHub Pages

工作流 [`.github/workflows/plan-board-pages.yml`](../../.github/workflows/plan-board-pages.yml) 会在 `main` 分支中的看板文件变化后，将本目录部署为 GitHub Pages。

首次使用时，需要在仓库 **Settings → Pages → Build and deployment** 中将 Source 选择为 **GitHub Actions**。成功后页面地址为：

```text
https://pto-isa.github.io/SuperNPUBench/
```
