- 编译 + qemu/gfrun 功能验证

## 参数解释:
- options:
  -h            show this help message and exit
  -src          src dir path, case: /xx/BenchMark/
  -m            test model: cmp or run, default cmp
  -llvm         llvm dir path, case: /xx/linx_blockisa_llvm
  -qemu         qemu-linx path
  -gfrun        gfrun path
  -i            elf dir, Directly input the binary package without compilation. elf name: xx.bin

- 选择模式
-m:
cmp or run, default cmp
cmp：仅编译
run: 编译 + 运行

- 选择验证模型
-qemu
-gfrun
给那个参数，就验证那个，两个都给就两个都验证

## 使用方式

### 输入源码和编译工具链测试

- 仅编译
python3 /xx/test.py -src /xx/TileOP_Lib/ -llvm /xx/linx_blockisa_llvm-v0.42_simt

- 编译+运行
python3 /xx/test.py -src /xx/TileOP_Lib/ -llvm /xx/linx_blockisa_llvm-v0.42_simt -m run -qemu /xx/build/qemu-linx

### 输入二进制测试
输入-i，则使用该模式

python3 /xx/test.py -i /xx/elf_dir/ -qemu /xx/build/qemu-linx