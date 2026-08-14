#!/usr/bin/env python3
"""R 组随机用例：逐步看数据从哪搬到哪，以及 GM 此刻应该是什么样。

两件事：

  1. 可视化    每一步相对上一条指令，数据的来源与去向；GM 各区当前应装什么
  2. 独立判据  同一套推导可以直接算出期望的 GM 字节，用 --check 校验任意 dump

第 2 点是重点。端到端比对一直以 gfrun 为黄金参考，而 gfrun 与 gfsim 共用
isa/ 解码器 —— 共用层错了两边一起错，比对照样 IDENTICAL（2026-08-14 的 ADDTPC
回归就是这么把整套用例伪装成全绿的）。这里的期望值只依赖用例源码与 ISA 的自识别
编码，不碰任何模型，因此能独立判死。

    python3 viz_random_case.py src/r1_random_seq_i32.cpp -o r1_flow.html
    python3 viz_random_case.py src/r1_random_seq_i32.cpp --check dump.bin

编码（见 tlsu_bench.hpp）：元素值 = (tag<<28) | ((row+1)<<16) | (col+1)。
每次搬运都是整块 8x128 的 1:1 拷贝，所以一个区装什么完全由"最后写入的数据源自
哪份图样"决定 —— 期望值因此可以只用一个 tag 表示，再展开成 1024 个字。
"""
import argparse
import json
import re
import struct
import sys

# 与 tlsu_bench.hpp / gen_random_case.py 保持一致
ROWS, COLS = 8, 128
TILE_WORDS = ROWS * COLS            # 1024
TAGV = {"s0": 0xA, "s1": 0xB, "s2": 0xC, "s3": 0xD}
TAGL = {0xA: "A", 0xB: "B", 0xC: "C", 0xD: "D"}


def parse(path):
    src = open(path, encoding="utf-8").read()
    ops = []
    for kind, a, b in re.findall(r"\n\s+(TLOAD|TSTORE)\((\w+),\s*(\w+)\);", src):
        # 参数次序不对称：TSTORE(内存, tile) 而 TLOAD(tile, 内存)
        ops.append({"k": "store", "t": b, "m": a} if kind == "TSTORE"
                   else {"k": "load", "t": a, "m": b})
    seed = re.search(r"种子\s*:\s*(\d+)", src)
    return ops, (seed.group(1) if seed else "?")


def replay(ops):
    """走一遍序列，给每步补上"从哪到哪"和该步之后的 GM 期望。

    tile 暂存这一层用的是源码里的变量名。真实硬件是 m/t/u 三个滚动寄存器堆，
    B.IOT 用相对回溯（#1 = 最近压入）寻址，与源码变量不是一一对应 —— 这里只
    关心数据流向，不声称它是架构寄存器。
    """
    tiles, regions = {}, {}
    for op in ops:
        if op["k"] == "load":
            m = op["m"]
            if m.startswith("s"):
                st = {"tag": TAGV[m], "depth": 0, "src": m}
            else:
                r = regions.get(m)
                st = ({"tag": r["tag"], "depth": r["depth"], "src": m} if r
                      else {"tag": None, "depth": 0, "src": m})
            tiles[op["t"]] = st
            op["carry"] = dict(st)
            op["raw"] = bool(m.startswith("r") and regions.get(m))
            op["frm"], op["to"] = m, op["t"]
            op["gm_changed"] = None            # TLOAD 不改 GM
        else:
            st = tiles.get(op["t"]) or {"tag": None, "depth": 0, "src": None}
            old = regions.get(op["m"])
            new = {"tag": st["tag"], "depth": st["depth"] + 1, "src": st["src"]}
            regions[op["m"]] = new
            op["carry"] = dict(new)
            op["frm"], op["to"] = op["t"], op["m"]
            op["origin"] = st["src"]           # 数据上一跳
            op["gm_changed"] = op["m"]
            op["gm_old"] = old["tag"] if old else None
    return ops, regions


def expected_bytes(regions, nreg=16):
    """把最终的 tag 映射展开成完整的 GM 字节。"""
    words = []
    for i in range(nreg):
        tag = (regions.get("r%d" % i) or {}).get("tag")
        if tag is None:
            words += [0] * TILE_WORDS
        else:
            words += [(tag << 28) | ((r + 1) << 16) | (c + 1)
                      for r in range(ROWS) for c in range(COLS)]
    return struct.pack("<%dI" % len(words), *words)


def check(path, dump_path):
    ops, _ = parse(path)
    ops, regions = replay(ops)
    exp = expected_bytes(regions)
    got = open(dump_path, "rb").read()
    if len(got) != len(exp):
        print(f"长度不符：期望 {len(exp)}，dump {len(got)}")
        return 1
    if exp == got:
        print(f"PASS  {dump_path} 与独立推导的期望值逐字节一致（{len(exp)} 字节）")
        return 0
    for i in range(0, len(exp), 4):
        if exp[i:i + 4] != got[i:i + 4]:
            e = struct.unpack("<I", exp[i:i + 4])[0]
            g = struct.unpack("<I", got[i:i + 4])[0]
            reg, off = divmod(i // 4, TILE_WORDS)
            print(f"FAIL  首处不一致 @字节 {i}（R{reg} 第 {off // COLS} 行 "
                  f"第 {off % COLS} 列）期望 {e:08x} 实际 {g:08x}")
            break
    return 1


PAGE = r"""<title>R 组 GM 演进</title>
<style>
:root{
  --paper:#F5F7F9; --panel:#FFFFFF; --ink:#16202B; --ink2:#4A5A6B; --ink3:#7E8E9E;
  --rule:#D5DCE3;
  --load:#1C6FB5; --store:#C25A22;
  --d1:#8CA9C4; --d2:#5D86AB; --d3:#3A6390; --d4:#1E4370;
  --empty:#DFE5EB;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#141A21; --panel:#1A222B; --ink:#DDE5EC; --ink2:#9CACBB; --ink3:#6E7E8D;
  --rule:#26313C;
  --load:#4593CF; --store:#D4753F;
  --d1:#4E6E8E; --d2:#6F92B0; --d3:#93B4CE; --d4:#B9D2E6;
  --empty:#232E39;
}}
:root[data-theme="dark"]{
  --paper:#141A21; --panel:#1A222B; --ink:#DDE5EC; --ink2:#9CACBB; --ink3:#6E7E8D;
  --rule:#26313C;
  --load:#4593CF; --store:#D4753F;
  --d1:#4E6E8E; --d2:#6F92B0; --d3:#93B4CE; --d4:#B9D2E6;
  --empty:#232E39;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;padding:38px 22px 60px}
.wrap{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
h1{font:600 31px/1.15 ui-serif,"Iowan Old Style",Palatino,Georgia,serif;margin:0 0 6px;
  letter-spacing:-.01em}
.eyebrow{font:500 11px/1 ui-monospace,"SF Mono",Menlo,monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink3);margin:0 0 9px}
.sub{color:var(--ink2);margin:0;max-width:70ch}
.stage{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px 8px}
svg{display:block;width:100%;height:auto}
#flow{min-width:700px}
.scroll{overflow-x:auto}
.ctl{display:flex;gap:9px;align-items:center;flex-wrap:wrap;background:var(--panel);
  border:1px solid var(--rule);border-radius:10px;padding:10px 13px}
button{font:500 13px/1 system-ui,sans-serif;color:var(--ink);background:var(--paper);
  border:1px solid var(--rule);border-radius:7px;padding:8px 12px;cursor:pointer}
button:hover{border-color:var(--ink3)}
button:focus-visible{outline:2px solid var(--load);outline-offset:2px}
button.pri{background:var(--load);border-color:var(--load);color:#fff}
input[type=range]{flex:1;min-width:170px;accent-color:var(--load)}
.step{font:600 14px ui-monospace,monospace;font-variant-numeric:tabular-nums;
  color:var(--ink2);min-width:84px}
.say{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--load);
  border-radius:8px;padding:12px 15px;font:13.5px/1.7 ui-monospace,"SF Mono",Menlo,monospace;
  min-height:86px}
.say.st{border-left-color:var(--store)}
.say b{color:var(--ink);font-weight:600}
.say .dim{color:var(--ink3)}
.mv{font-size:15px;letter-spacing:.02em}
.node{fill:var(--empty);stroke:var(--rule);stroke-width:1}
.node.d1{fill:var(--d1)} .node.d2{fill:var(--d2)}
.node.d3{fill:var(--d3)} .node.d4{fill:var(--d4)}
.node.src{fill:none;stroke:var(--ink3);stroke-width:1.2;stroke-dasharray:3 2.5}
.node.from{stroke:var(--load);stroke-width:2.6}
.node.to{stroke:var(--store);stroke-width:2.6}
.nm{font:600 11.5px ui-monospace,monospace;dominant-baseline:middle;pointer-events:none}
.nm.on{fill:var(--paper)} .nm.off{fill:var(--ink2)}
.hex{font:9.5px ui-monospace,monospace;pointer-events:none}
.hex.on{fill:var(--paper);opacity:.9} .hex.off{fill:var(--ink3)}
.collab{font:500 10px ui-monospace,monospace;letter-spacing:.11em;fill:var(--ink3);
  text-transform:uppercase}
.edge{fill:none;stroke-width:2.6;stroke-linecap:round}
.edge.load{stroke:var(--load)} .edge.store{stroke:var(--store)}
.tlmk.load{fill:var(--load)} .tlmk.store{fill:var(--store)}
.tlmk.future{opacity:.2}
.head{stroke:var(--ink);stroke-width:1.6}
.legend{display:flex;gap:17px;flex-wrap:wrap;align-items:center;font-size:12.5px;color:var(--ink2)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;
  vertical-align:-1px}
.legend .ln{width:20px;height:2.5px;border-radius:2px}
.ramp{display:inline-flex;gap:2px;margin-right:6px;vertical-align:-1px}
.ramp i{width:13px;height:11px;border-radius:2px;margin:0}
.oracle{border-left:3px solid var(--store);padding:2px 0 2px 16px;display:flex;
  flex-direction:column;gap:8px}
.oracle p{margin:0;max-width:74ch;font-size:13.5px;color:var(--ink2)}
.oracle strong{color:var(--ink)}
code{font:12.5px ui-monospace,monospace;background:var(--paper);padding:1px 5px;
  border-radius:4px;border:1px solid var(--rule)}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">__CASE__ · seed __SEED__</p>
  <h1>R 组 GM 演进</h1>
  <p class="sub">逐步推进 __N__ 个块，看每一条指令把数据从哪搬到了哪，以及此刻
  <strong>全局内存应该是什么样</strong>。期望值由用例源码与自识别编码独立推导，
  不依赖任何模型。</p>
</header>

<div class="ctl">
  <button id="first" title="回到开头">⏮</button>
  <button id="prev" title="上一步">◀</button>
  <button id="play" class="pri">▶ 播放</button>
  <button id="next" title="下一步">▶</button>
  <button id="last" title="跳到结尾">⏭</button>
  <span class="step" id="stepno">0 / __N__</span>
  <input type="range" id="scrub" min="0" max="__NMAX__" value="0">
  <button id="nstore">下一次 GM 变化</button>
  <button id="nraw">下一个 RAW</button>
  <button id="speed">速度 1×</button>
</div>

<div class="say" id="say"></div>

<div class="stage scroll"><svg id="flow" viewBox="0 0 1000 560" role="img"
  aria-label="源图样、tile 暂存与 16 个 GM 区域的当前内容"></svg></div>

<div class="legend">
  <span><i class="ln" style="background:var(--load)"></i>来源</span>
  <span><i class="ln" style="background:var(--store)"></i>去向</span>
  <span><span class="ramp"><i style="background:var(--d1)"></i><i style="background:var(--d2)"></i><i style="background:var(--d3)"></i><i style="background:var(--d4)"></i></span>转手 1 → 4+ 次</span>
  <span><i style="background:var(--empty)"></i>还没被写过（全零）</span>
</div>

<div class="stage scroll">
  <svg id="tl" viewBox="0 0 1000 58" style="min-width:700px" role="img"
    aria-label="时间条，竖线是当前位置"></svg>
</div>

<section>
  <p class="eyebrow">这张图同时是一个独立判据</p>
  <div class="oracle">
    <p>元素值 = <code>(tag&lt;&lt;28) | ((row+1)&lt;&lt;16) | (col+1)</code>，每次搬运都是整块
    8×128 的 1:1 拷贝 —— 所以一个区最终装什么，<strong>完全由"最后写入的数据源自哪份
    图样"决定</strong>，可以脱离模型直接算出全部 65536 字节。</p>
    <p>这一点很要紧：端到端比对一直以 gfrun 为黄金参考，而 gfrun 与 gfsim
    <strong>共用 <code>isa/</code> 解码器</strong> —— 共用层错了两边一起错，逐字节比对照样
    IDENTICAL。2026-08-14 的 ADDTPC 回归正是这样把整套用例伪装成全绿的。</p>
    <p>校验任意 dump：<code>python3 viz_random_case.py src/__CASE__.cpp --check dump.bin</code></p>
  </div>
</section>
</div>

<script>
const OPS = __OPS__, N = OPS.length;
const SRC = __SRC__, REG = __REG__, TIL = __TIL__;
const TAGL = {10:"A",11:"B",12:"C",13:"D"};
const SVG = "http://www.w3.org/2000/svg";
const el = (t,a)=>{const e=document.createElementNS(SVG,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
const up = s => s.toUpperCase();

/* 期望的首字与末字：row0col0 与 row7col127 */
const w0 = tag => tag==null ? "00000000"
  : ((tag<<28)>>>0 | (1<<16) | 1).toString(16).padStart(8,"0");
const wN = tag => tag==null ? "00000000"
  : ((tag<<28)>>>0 | (8<<16) | 128).toString(16).padStart(8,"0");

/* ── 几何：源图样一行、tile 暂存一行、GM 区域 4x4 ── */
const pos = {};
SRC.forEach((n,i)=>pos[n]={x:60+i*150, y:52, w:120, h:34, kind:"s"});
TIL.forEach((n,i)=>pos[n]={x:44+i*118, y:136, w:100, h:30, kind:"t"});
REG.forEach((n,i)=>{const c=i%4, r=(i/4)|0;
  pos[n]={x:36+c*242, y:238+r*76, w:222, h:60, kind:"r"};});

function stateAt(step){
  const tiles={}, regions={};
  for(let i=0;i<=step&&i<N;i++){const o=OPS[i];
    if(o.k==="load") tiles[o.t]=o.carry; else regions[o.m]=o.carry;}
  return {tiles,regions};
}

const flow = document.getElementById("flow");
function box(a,b){ // 两个节点中心之间的曲线
  const p=pos[a], q=pos[b];
  const x1=p.x+p.w/2, y1=p.y+p.h, x2=q.x+q.w/2, y2=q.y;
  const my=(y1+y2)/2;
  return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
}

function draw(step){
  const S=stateAt(step), cur=(step>=0&&step<N)?OPS[step]:null;
  flow.textContent="";
  [["源图样 只读",44,30],["tile 暂存 源码级 t0–t7",30,116],["GM 目的区 判据",30,220]]
    .forEach(([t,x,y])=>flow.appendChild(Object.assign(
      el("text",{x,y,class:"collab"}),{textContent:t})));

  if(cur) flow.appendChild(el("path",{d:box(cur.frm,cur.to),class:"edge "+cur.k}));

  const node=(n)=>{
    const p=pos[n], st=p.kind==="t"?S.tiles[n]:p.kind==="r"?S.regions[n]:null;
    const d=st?st.depth:0, tag=p.kind==="s"?(10+ +n.slice(1)):(st?st.tag:null);
    let cls="node"+(p.kind==="s"?" src":(d?" d"+Math.min(d,4):""));
    if(cur&&n===cur.frm) cls+=" from"; if(cur&&n===cur.to) cls+=" to";
    const g=el("g",{});
    g.appendChild(el("rect",{x:p.x,y:p.y,width:p.w,height:p.h,rx:6,class:cls}));
    const lit=p.kind!=="s"&&d>0;
    g.appendChild(Object.assign(el("text",{x:p.x+10,y:p.y+(p.kind==="r"?18:p.h/2),
      class:"nm "+(lit?"on":"off")}),{textContent:up(n)+(tag!=null?"　图样 "+TAGL[tag]:"")}));
    if(p.kind==="r"){
      g.appendChild(Object.assign(el("text",{x:p.x+10,y:p.y+36,
        class:"hex "+(lit?"on":"off")}),{textContent:"0x"+w0(tag)+" … 0x"+wN(tag)}));
      g.appendChild(Object.assign(el("text",{x:p.x+10,y:p.y+50,
        class:"hex "+(lit?"on":"off")}),{textContent:
          st?`第 ${st.depth} 手，上一跳 ${up(st.src)}`:"从未写入"}));
    }
    flow.appendChild(g);
  };
  SRC.forEach(node); TIL.forEach(node); REG.forEach(node);
  narrate(step,S); timeline(step);
}

const say=document.getElementById("say");
function narrate(step,S){
  if(step<0||step>=N){say.innerHTML="<span class='dim'>序列结束。</span>";return;}
  const o=OPS[step], c=o.carry;
  say.className="say"+(o.k==="store"?" st":"");
  let s=`<div class="mv"><b>块 ${step}</b>　`;
  if(o.k==="load"){
    s+=`<b>${up(o.frm)} → ${up(o.to)}</b>　<span class="dim">TLOAD</span></div>`;
    s+=o.frm[0]==="s"
      ? `从只读图样 ${TAGL[c.tag]} 取一块（第一手）`
      : (c.tag!=null
          ? `读回 ${up(o.frm)}，里面是图样 ${TAGL[c.tag]}，已转手 ${c.depth} 次`
            +(o.raw?`　<b>← RAW：必须看见之前对 ${up(o.frm)} 的写</b>`:"")
          : `<span class="dim">${up(o.frm)} 还没被写过，读到的是零</span>`);
    s+=`<br><span class="dim">GM 不变 —— TLOAD 只把数据取进 tile</span>`;
  }else{
    s+=`<b>${up(o.frm)} → ${up(o.to)}</b>　<span class="dim">TSTORE</span></div>`;
    s+=`${up(o.to)} 现在装图样 ${TAGL[c.tag]||"零"}，第 ${c.depth} 手`;
    s+=o.origin&&o.origin[0]==="r"
      ? `　<b>← 数据上一跳来自 ${up(o.origin)}</b>`
      : `　<span class="dim">数据直接来自 ${up(o.origin||"?")}</span>`;
    s+=`<br><b>GM 变化：</b>${up(o.to)}　`
      +`${o.gm_old!=null?"图样 "+TAGL[o.gm_old]:"全零"} → 图样 ${TAGL[c.tag]||"零"}`
      +`　<span class="dim">0x${w0(c.tag)} … 0x${wN(c.tag)}</span>`;
  }
  say.innerHTML=s+`<br><span class="dim">已写过 ${Object.keys(S.regions).length}/16 个区</span>`;
}

const tl=document.getElementById("tl");
function timeline(step){
  tl.textContent="";
  const L=8, plot=984, x=i=>L+(i+.5)/N*plot;
  for(let i=0;i<N;i++){const o=OPS[i], h=o.k==="load"?20:28;
    tl.appendChild(el("rect",{x:x(i)-1.5,y:32-h,width:3,height:h,rx:1.3,
      class:"tlmk "+o.k+(i>step?" future":"")}));}
  tl.appendChild(el("line",{x1:x(step),y1:2,x2:x(step),y2:38,class:"head"}));
  for(let t=0;t<=N;t+=25)
    tl.appendChild(Object.assign(el("text",{x:L+t/N*plot,y:52,
      style:"fill:var(--ink3);font:10.5px ui-monospace,monospace;text-anchor:middle"}),
      {textContent:t}));
}

let cur=0,timer=null,speed=1;
const scrub=document.getElementById("scrub"), stepno=document.getElementById("stepno");
function go(i){cur=Math.max(0,Math.min(N-1,i));scrub.value=cur;
  stepno.textContent=`${cur} / ${N}`;draw(cur);}
const jump=pred=>{stop();
  for(let i=cur+1;i<N;i++) if(pred(OPS[i])) return go(i);
  for(let i=0;i<N;i++) if(pred(OPS[i])) return go(i);};
document.getElementById("first").onclick=()=>{stop();go(0);};
document.getElementById("prev").onclick=()=>{stop();go(cur-1);};
document.getElementById("next").onclick=()=>{stop();go(cur+1);};
document.getElementById("last").onclick=()=>{stop();go(N-1);};
document.getElementById("nstore").onclick=()=>jump(o=>o.k==="store");
document.getElementById("nraw").onclick=()=>jump(o=>o.raw);
scrub.oninput=e=>{stop();go(+e.target.value);};
const playBtn=document.getElementById("play");
function stop(){if(timer){clearInterval(timer);timer=null;playBtn.textContent="▶ 播放";}}
playBtn.onclick=()=>{if(timer)return stop();
  playBtn.textContent="❚❚ 暂停";
  timer=setInterval(()=>{if(cur>=N-1){stop();return;}go(cur+1);},640/speed);};
document.getElementById("speed").onclick=e=>{
  speed=speed===1?2:speed===2?4:1;e.target.textContent=`速度 ${speed}×`;
  if(timer){stop();playBtn.click();}};
addEventListener("keydown",e=>{
  if(e.key==="ArrowRight"){stop();go(cur+1);}
  else if(e.key==="ArrowLeft"){stop();go(cur-1);}
  else if(e.key===" "){e.preventDefault();playBtn.click();}});
go(0);
</script>
"""


def build(path, out):
    ops, seed = parse(path)
    ops, regions = replay(ops)
    srcs = sorted({o["m"] for o in ops if o["m"].startswith("s")}, key=lambda s: int(s[1:]))
    regs = sorted({o["m"] for o in ops if o["m"].startswith("r")}, key=lambda s: int(s[1:]))
    tils = sorted({o["t"] for o in ops}, key=lambda s: int(s[1:]))
    case = path.split("/")[-1].rsplit(".", 1)[0]

    html = (PAGE
            .replace("__OPS__", json.dumps(ops, ensure_ascii=False, separators=(",", ":")))
            .replace("__SRC__", json.dumps(srcs))
            .replace("__REG__", json.dumps(regs))
            .replace("__TIL__", json.dumps(tils))
            .replace("__CASE__", case)
            .replace("__SEED__", seed)
            .replace("__NMAX__", str(len(ops) - 1))
            .replace("__N__", str(len(ops))))
    open(out, "w", encoding="utf-8").write(html)

    r2r = sum(1 for o in ops if o.get("origin", "").startswith("r") and o["origin"] != o["m"])
    raw = sum(1 for o in ops if o.get("raw"))
    print(f"{path}: {len(ops)} 步 / {raw} 个 RAW / {r2r} 次区域→区域搬运 -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", help="生成的用例 .cpp")
    ap.add_argument("-o", "--output", help="输出 html")
    ap.add_argument("--check", metavar="DUMP", help="用独立推导的期望值校验一个内存 dump")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.case, args.check))
    if not args.output:
        ap.error("需要 -o 或 --check")
    build(args.case, args.output)
