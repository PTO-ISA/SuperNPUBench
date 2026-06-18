- 编译-src指令目录（递归）的所有.cpp文件
- jcore：编译，qemu/gfrun的运行 + 功能验证
- cpu_sim：编译 + 运行

## 参数解释:
- options:
  -h            show this help message and exit
  -lib          TileOP库的根目录， case: /xx/PTOTileLib/
  -src          需要编译的目录（递归的）, case: /xx/PTOTileLib/test/tileop_api/src/
                默认等于lib
  -m            test model: cmp or run, default cmp
  -lc           linx clang++ path, case: /xx/linx_blockisa_llvm/bin/clang++
  -hc           cpu_sim clang++ path, case: /xx/llvm-15.0.4/bin/clang++
  -qemu         qemu-linx path
  -gfrun        gfrun path
  -case         单一测例测试时的测例名，缺省时运行所有测例, case: TAdd

- 选择模式
-m:
cmp or run, default cmp
cmp：仅编译
run: 编译 + 运行

- 选择验证模型
-qemu
-gfrun
给那个参数，就验证那个，两个都给就两个都验证

## 使用实例

- 编译 cpu_sim版本
python3 /xx/test.py -lib /xx/PTOTileLib/ -src /xx/PTOTileLib/test/tileop_api/src -hc /xx/llvm-15.0.4/bin/clang++

- 编译 jcore版本
python3 /xx/test.py -lib /xx/PTOTileLib/ -src /xx/PTOTileLib/test/tileop_api/src -lc /xx/linx_blockisa_llvm/bin/clang++

- 编译+运行 cpu_sim版本
python3 /xx/test.py -lib /xx/PTOTileLib/ -src /xx/PTOTileLib/test/tileop_api/src -hc /xx/llvm-15.0.4/bin/clang++ -m run

- 编译+运行+功能验证 jcore版本
python3 /xx/test.py -lib /xx/PTOTileLib/ -src /xx/PTOTileLib/test/tileop_api/src -lc /xx/linx_blockisa_llvm/bin/clang++ -hc /xx/llvm-15.0.4/bin/clang++ -qemu /xx/qemu-linx -m run

- 编译+运行+功能验证 单用例
python3 {$REPO}/test/tileop_api/test_tileop.py -lib {$REPO} -src {$REPO}/test/tileop_api/src -lc {$L_CHAIN}/bin/clang++ -hc {$H_CHAIN}/bin/clang++ -qemu {$QEMU}/qemu-linx -m run -case=Txxx