#!/usr/bin/env python3
"""构建固定评测集（Spec 10.5 / 阶段 0）。

数据来源：
- 现有 40 条：从 evaluation_data/ragas_eval_dataset_filled.json 拷贝
  question/ground_truth，并按下方 TYPE_MAP 逐条标注 question_type。
- 新增 51 条：evaluation_data/_part_a.json（terminology/formula_table）、
  _part_b.json（summary/comparison）、_part_c.json（multi_hop/unanswerable），
  由人工+子代理基于 datas/abacus.md 真实内容编写（unanswerable 类除外）。

输出 evaluation_data/fixed_eval_dataset.json，并校验分层下限：
factual≥15, terminology≥10, summary≥10, comparison≥10,
multi_hop≥10, formula_table≥10, unanswerable≥10。
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation_data"

# 现有 40 条（按 ragas_eval_dataset_filled.json 中的顺序）的类型标注
TYPE_MAP: dict[int, str] = {
    0: "factual",        # KPT Line 模式 + symmetry=0
    1: "factual",        # Atomkit 5 个参数含义
    2: "factual",        # nscf 三参数配置
    3: "formula_table",  # BANDS_1.dat 各列数据含义
    4: "factual",        # 铁磁性两项输入设置
    5: "factual",        # out_dos=2 输出 PDOS
    6: "terminology",    # PDOS <orbital> 标签 l/m/z 属性
    7: "factual",        # abacus-plot efermi 来源
    8: "factual",        # 电荷密度/波函数 cube 格式与可视化
    9: "factual",        # bands_to_print 核心参数
    10: "multi_hop",     # LCAO 能带分解电荷密度需两步执行
    11: "comparison",    # if_separate_k true/false 输出区别
    12: "factual",       # 能带分解电荷密度为何关闭对称性
    13: "terminology",   # ELF 取值范围与物理意义
    14: "factual",       # 自旋极化 ELF cube 文件
    15: "factual",       # ELF χ 分子微小修正
    16: "factual",       # Bader 分析前 INPUT 设置
    17: "factual",       # 自旋极化不能分别做 Bader 的原因
    18: "multi_hop",     # 自旋极化 Bader 完整操作流程
    19: "formula_table", # ACF.dat 净电荷计算
    20: "factual",       # 弹性常数应变构型数量
    21: "factual",       # 弹性常数 calculation 类型
    22: "factual",       # phonopy -d 命令作用
    23: "factual",       # 声子谱 INPUT calculation/cal_force
    24: "factual",       # 二阶力常数输出设置
    25: "formula_table", # FORCE_CONSTANTS 单位差异与转换
    26: "factual",       # 三阶力常数不能用 dpdata 的原因
    27: "comparison",    # Wannier90 对两种基组的支持状态
    28: "factual",       # Wannier90 两个输出参数
    29: "factual",       # Candela 四类核心性质
    30: "factual",       # USPEX OMP_NUM_THREADS
    31: "factual",       # Hefei-NAMD MD 两个输出参数
    32: "factual",       # pyabacus 五个核心模块
    33: "factual",       # hsolver 三种对角化算法
    34: "factual",       # C++ 命名规则
    35: "factual",       # Doxygen 三种注释格式
    36: "factual",       # v3.7.0 输入参数存储方式变化
    37: "factual",       # 异构计算代码子目录
    38: "factual",       # formatter-2.0 三部分功能
    39: "factual",       # .clang-format 依赖版本
}

STRATA_MIN = {
    "factual": 15,
    "terminology": 10,
    "summary": 10,
    "comparison": 10,
    "multi_hop": 10,
    "formula_table": 10,
    "unanswerable": 10,
}


def main() -> int:
    filled = json.loads(
        (EVAL_DIR / "ragas_eval_dataset_filled.json").read_text(encoding="utf-8")
    )
    assert len(filled) == len(TYPE_MAP), "现有评测集条数与 TYPE_MAP 不一致"

    dataset: list[dict] = []
    for idx, item in enumerate(filled):
        dataset.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "question_type": TYPE_MAP[idx],
                "source_hint": "继承自 ragas_eval_dataset_filled.json（基于 datas/abacus.md）",
            }
        )

    for part in ("_part_a.json", "_part_b.json", "_part_c.json"):
        entries = json.loads((EVAL_DIR / part).read_text(encoding="utf-8"))
        dataset.extend(entries)

    # 校验字段完整性
    required = {"question", "ground_truth", "question_type"}
    for i, entry in enumerate(dataset):
        missing = required - set(entry)
        assert not missing, f"第 {i} 条缺字段: {missing}"

    counts = Counter(e["question_type"] for e in dataset)
    print(f"总条数: {len(dataset)}")
    ok = True
    for qtype, minimum in STRATA_MIN.items():
        n = counts.get(qtype, 0)
        status = "OK" if n >= minimum else "不足!"
        if n < minimum:
            ok = False
        print(f"  {qtype:<15} {n:>3} (下限 {minimum}) {status}")

    if not ok:
        print("分层下限未满足，不写出文件", file=sys.stderr)
        return 1

    out = EVAL_DIR / "fixed_eval_dataset.json"
    out.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
