# FA model composition and limitations

Required assets are publicly published and their actual downloaded bytes were verified. 4/4 actual downloaded cases reproduced rc0 and historical cycles; zero-input timing only. Actual model source/build/payload identities are recorded in dependency/; original operator dirty/API/ISA and numerical gaps remain. Streamed JSON is a host observability/memory mechanism, not the claimed cycle gain. Historical four-case reproduction is not a clean-main model comparison.

| 修改 / 分类 | 目的、commit、文件 | 已验证 / 未闭合 / 上游状态 |
|---|---|---|
| raw TPACK + Local parent lifecycle / 功能与时序修复 | `dd86f8a20b9161bdb1f836dc234c86d187d561e4`；`TimingSim/pe/vec/{VecCompute.h,VecTileSplit.cpp,VecValuInfo.h}`、`TimingSim/frontend/bctrl/BIssue.cpp`、`TimingSim/common/ModelCommon/BlockCommand.cpp`；raw field拼接、valid prefix、source subview、physical CELL切分与Shared路由，INIT查旧parent/窄writer descriptor生命周期修复 | issue1010专项100/100、gfsim19/19；前端同ELF544B独立golden及负控通过；不是完整FA数值。#1010 closed，PR1057 merged 2026-10-24，merge `6d4ea35ec8029596882a245316c2f30b8dab9314` |
| retire drain / 资源生命周期正确性修复 | `ec578433e46cdd49c4a6c8c0ae9d39a546ecd409`；`TimingSim/frontend/bctrl/BIssue.cpp`及回归；乱序退休后排空可回收vtag前缀；不增加容量、不更改ASSEMBLE语义；与TPACK合成`15c5898d0b3008b3a2e4ebc6eb4f631cc413bbe1` | 专项8/8、timing451/451、gfsim19/19；旧64K无rename断言142143仅checkoff，非独立数值。#1053 closed，PR1054 merged 2026-10-23，merge `32c04e840e7cb40fcdc9247cb569af5d5473598a` |
| stream complete JSON / 可观测性改动 | `89b2366ea704fd75fd702440496439dadf5496be`在`15c5898d0b3008b3a2e4ebc6eb4f631cc413bbe1`上；`TimingSim/infra/SimSys.cpp`、`tools/trace_logger/TraceLogger.{cpp,h}`、`tests/timing_test/trace_stream_test.cpp`；不聚合完整event DOM，保留完整JSON | 3/3 streaming、467/467 timing；该commit本身不是新的性能优化或FA算法数值修复；未识别专门上游PR，不推测 |
| API TPACK emit / 接口层，不能混成model修复 | API baseline `ecae381331e2dd4ef13270df5518ca9ebda19ef4`，upstream TPACK `8f43777b020bc6e8114ad70f35c2751deddc0ab9`；NPUBench归档`dependencies/api_tpack_emit.patch` | 保存Local MIDDLE/LAST parent生命周期和subview offset；168文件历史hash核对无差异。原20519 ISA revision仍未建立，不能据新版ASL回填原build合同 |
| BroadcastByteOffset / 仅数值背景线索 | #957（gfrun PR）、#926、#979；属于TROWEXPAND广播offset/ASL背景，**不是上面TPACK或retire组合的已证明根因** | 当前957 open未合入，926 closed，979 open；完整FA6756/32768超容差且旧新compiler输出byte-identical，不能据issue状态宣称修复或因果闭合 |

FA原证据：`perf_analysis/issue1010_tpack_20261021/README.md`；`perf_analysis/algod_exploration_20261015/cellreg_clean_fix_20261020/issue1053_fix_20261021/README.md`；`perf_analysis/algod_exploration_20261015/algod2_packed_rowsum_20261021/skv64k/{model,trace_export_model}/CLOSED_MODEL.md`；归档`split_20261109/ndf-fa-20261109-r2/{dependencies.json,evidence/*/run.json}`。共享恢复的实际7文件patch见`dependency/portable-model.patch`；历史本地文件名仅是溯源标签，不作为重放依赖。



Fixed old metadata index: https://github.com/LinxISA/SuperScalarModel/tree/b28d6865a84c91b650eee779e4bf5280381a1d65/archives/ndf-fa-20261109-r2/

Online issue snapshot: issue_status.json. Local paths in historical narrative are provenance labels, not executable reproduction dependencies; obtain required artifacts through release_assets.json; actual delivery and replay receipts are recorded separately.

## Portable recovery and build snapshot

Download/extract matching runtime Release asset from release_assets.json. Fetch public Model branch fix/gfsim-tpack-issue1010; then python3 <runtime>/rebuild_model.py --model-repo <clone-containing-dd86f8a20b9161bdb1f836dc234c86d187d561e4> --source-out <new-source> --build-out <new-build>. Recipe archives fixed public base and applies source/portable-model.patch once; resulting tree424a94d7c5ae79f4584234298b99a65df8d4f1fb. No compiler/toolchain is needed for prebuilt replay.

See dependency/public_source_recovery.json, dependency/model_build_toolchain.json and dependency/source_compile_flags.json. Public base plus patch matches historical model tree exactly; source archive3347 blobs matches Git with no missing/extra/mismatched files. The rebuilt runtime hash is `f288a7053ddeeafabf42df45b292d04b47954022c6c5c1d990bde633ce40447a`; original runtime remains `938fc03780f38bee1aa012caa6b7fc53b2acfb6f504a2ef647a422a1f0304544`. These are distinct identities and the portable result ids are new.
