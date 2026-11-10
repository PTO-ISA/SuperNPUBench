# source/ —— 编译这颗 ELF 的那一版源码

这是 **unscoped D2** 变体的四文件快照（`fa_lowp_algD2.hpp` = `1bc9534a…`，28,523 B），构建宏含
`-DFA_ALGD2=1` 与 `-DFA_ALGB_D_DBLQ=1`。它与 `operator_identity.json` 中记录的哈希逐字节一致，
也与 `fa-operator-r2.tar.gz` 内 `source_snapshot/fa-64k-unscoped-d2-258099/` 相同。

**不要与姊妹目录混淆**：`../fa-scoped-packed-rowsum-20261109/source/` 存的是 final scoped D2
（`fa_lowp_algD2.hpp` = `203a0ee7…`，28,799 B），**不是**这颗 ELF 的来源；按那份重建会得到另一个二进制。

本目录只用于保证"能拿到当时编译的源码"。构建配方见 `../build/recipe.json`；本包**没有**重建 ELF
（工具链二进制未随包），因此 `operator_build` 状态是 `not_checked`。

| 文件 | 字节 | sha256 |
|---|---:|---|
| `fa_lowp_algD_driver.cpp` | 3,582 | `05b6d4881c8f346a47fe6d8676d4a0dca76e2beb7874e0ef75d841c092db34d7` |
| `multi_thread/fa_lowp_algB/fa_lowp_algD.hpp` | 27,581 | `a30a523ddb0faec90b2b00c96d78ef61cfaabe17877f08fe1ac1b4cc9c885b82` |
| `multi_thread/fa_lowp_algB/fa_lowp_algD2.hpp` | 28,523 | `1bc9534a2fa0a45a86fd4a75f1c33dd205091469000a38b26c44972566891125` |
| `multi_thread/fa_lowp_algB/fa_lowp_algB.hpp` | 41 | `7670a8e0f6a4f0f20c5c998ecfbeb0ad3ebbfad7040e44f4b134f9cce83f46da` |

## 许可与来源

源码来自算子 Release 资产 `fa-operator-r2.tar.gz`（同一仓库、tag `work-archive-fa-20261109`），
本目录只是把它放进 git 树以便追溯。版权与使用条件以该资产及算子仓为准。