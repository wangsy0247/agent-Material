"""Materials Project 材料下载器 — 自包含，零项目依赖。"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

# numpy 必须在 pymatgen 之前导入，避免循环导入
import numpy  # noqa: F401
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)

# 配置（从环境变量读取，与 Agent 项目零依赖）
MP_API_KEY: str = os.environ.get("MP_API_KEY", "")
DOWNLOAD_DIR: str = os.environ.get("MP_DOWNLOAD_DIR", "./downloads")
# HISTORY_FILE: str = os.environ.get("MP_HISTORY_FILE", "./downloads/history.txt")

_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._+-]+")


def _safe_path_component(value: object, *, fallback: str) -> str:
    """把外部 API 或用户输入转换为不可越界的目录名。"""
    component = _UNSAFE_PATH_CHARS.sub("-", str(value)).strip(".-")[:80]
    return component or fallback


class MaterialsProjectDownloader:
    """封装 Materials Project 材料下载、去重与记录逻辑。"""

    def __init__(self):
        self.api_key = MP_API_KEY
        self.workspace_dir = DOWNLOAD_DIR
        self.time_str = datetime.now().strftime("%m")
        # self.history_file = HISTORY_FILE
        os.makedirs(self.workspace_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 历史记录
    # ------------------------------------------------------------------

    # def get_downloaded_formulas(self) -> set:
        # if not os.path.exists(self.history_file):
        #     return set()
        # formulas = set()
        # with open(self.history_file, "r", encoding="utf-8") as f:
        #     for line in f:
        #         if line.strip():
        #             formulas.add(line.split(" (")[0].strip())
        # return formulas

    # def is_duplicate(self, formula: str) -> bool:
    #     return formula in self.get_downloaded_formulas()

    # def record_download(self, formula: str, mp_id: str) -> None:
    #     with open(self.history_file, "a", encoding="utf-8") as f:
    #         f.write(f"{formula} ({mp_id})\n")

    # ------------------------------------------------------------------
    # 属性查询
    # ------------------------------------------------------------------

    def query_material_properties(self, formula: str) -> dict | None:
        if not self.api_key:
            return None

        try:
            mpr = MPRester(self.api_key)

            # 1. 批量获取所有匹配材料的摘要，按形成能排序（2 次 API 调用）
            summary_fields = [
                "material_id",
                "formation_energy_per_atom",
                "energy_above_hull",
                "band_gap",
                "density",
                "symmetry",
                "is_magnetic",
                "bulk_modulus",
            ]
            summary_docs = list(
                mpr.materials.summary.search(formula=formula, fields=summary_fields)
            )
            if not summary_docs:
                return None

            def _formation_energy(s):
                v = s.formation_energy_per_atom
                return v if v is not None else float("inf")

            summary_docs.sort(key=_formation_energy)
            best_summary = summary_docs[0]

            # 2. 仅获取最稳定材料的结构
            structure_docs = mpr.materials.search(
                material_ids=[best_summary.material_id],
                fields=["material_id", "structure"],
            )
            if not structure_docs:
                return None

            best_doc = structure_docs[0]
            mp_id = best_doc.material_id
            structure = best_doc.structure

            # 3. 提取晶体结构信息
            sga = SpacegroupAnalyzer(structure)
            conv_struct = sga.get_conventional_standard_structure()
            lattice = conv_struct.lattice
            a, b, c = [round(x, 3) for x in lattice.abc]
            alpha, beta, gamma = [round(x, 3) for x in lattice.angles]

            # 4. 提取物理属性
            try:
                k_modulus = (
                    round(best_summary.bulk_modulus.get("vrh", 0), 2)
                    if best_summary.bulk_modulus
                    else "未知"
                )
            except Exception:
                k_modulus = "未知"

            e_hull = (
                round(best_summary.energy_above_hull, 3)
                if best_summary.energy_above_hull is not None
                else "未知"
            )

            best_energy = best_summary.formation_energy_per_atom

            return {
                "Formula": formula,
                "MP_ID": mp_id,
                "Structure": structure,
                "Number of Atoms": conv_struct.num_sites,
                "Volume (Å³)": round(lattice.volume, 3),
                "Lattice a,b,c (Å)": f"{a}, {b}, {c}",
                "Angles α,β,γ (°)": f"{alpha}, {beta}, {gamma}",
                "Crystal System": (
                    best_summary.symmetry.crystal_system.value
                    if best_summary.symmetry
                    else "未知"
                ),
                "Space Group": (
                    best_summary.symmetry.symbol
                    if best_summary.symmetry
                    else "未知"
                ),
                "Density (g/cm³)": (
                    round(best_summary.density, 3)
                    if best_summary.density is not None
                    else "未知"
                ),
                "Band Gap (eV)": (
                    round(best_summary.band_gap, 3)
                    if best_summary.band_gap is not None
                    else "未知"
                ),
                "Is Magnetic": best_summary.is_magnetic,
                "Formation Energy (eV/atom)": (
                    round(best_energy, 3) if best_energy != float("inf") else "未知"
                ),
                "Energy Above Hull (eV/atom)": e_hull,
                "Bulk Modulus (GPa)": k_modulus,
            }

        except Exception as e:
            logger.error(f"查询 {formula} 属性时发生异常: {e}")
            return None

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def _save_files(self, formula: str, properties: dict) -> str:
        structure = properties.pop("Structure")
        workspace = Path(self.workspace_dir).resolve()
        safe_formula = _safe_path_component(formula, fallback="material")
        safe_mp_id = _safe_path_component(properties.get("MP_ID"), fallback="unknown")
        save_dir = (
            workspace / f"dataset{self.time_str}" / f"{safe_formula}_{safe_mp_id}"
        ).resolve()
        try:
            save_dir.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("下载目录越界") from exc
        save_dir.mkdir(parents=True, exist_ok=True)

        poscar_path = save_dir / "POSCAR"
        sga = SpacegroupAnalyzer(structure)
        conv_struct = sga.get_conventional_standard_structure()
        conv_struct.to(fmt="poscar", filename=str(poscar_path))

        json_path = save_dir / "properties.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(properties, f, ensure_ascii=False, indent=4)

        return str(save_dir)

    def download(self, formula: str) -> str:
        if not self.api_key:
            return "❌ 未配置 Materials Project API Key，请设置环境变量 MP_API_KEY。"

        # if self.is_duplicate(formula):
        #     return f"⚠️ 材料 {formula} 已存在于本地历史记录中，跳过下载。"

        properties = self.query_material_properties(formula)
        if properties is None:
            return f"❌ 在 Materials Project 中未找到化学式为 {formula} 的材料。"

        mp_id = properties["MP_ID"]
        save_dir = self._save_files(formula, properties)
        # self.record_download(formula, mp_id)

        logger.info(f"成功下载 {formula} (MP_ID: {mp_id}) → {save_dir}")
        return f"✅ 成功下载 {formula} (MP_ID: {mp_id})。文件已保存在: {save_dir}"

    # def list_history(self) -> str:
    #     if not os.path.exists(self.history_file):
    #         return "📭 本地暂无已下载的材料记录。"

    #     with open(self.history_file, "r", encoding="utf-8") as f:
    #         history = [line.strip() for line in f if line.strip()]

    #     if not history:
    #         return "📭 本地暂无已下载的材料记录。"

    #     return "📋 已下载材料列表（共 {} 条）：\n".format(len(history)) + "\n".join(
    #         history
    #     )
