import json
import os
import re
from datetime import datetime

from langchain_core.tools import tool
from mp_api.client import MPRester
from pydantic import BaseModel, Field
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from dotenv import load_dotenv

from utils.config_handler import agent_config, mp_config
from utils.logger import logger
from utils.path_tools import get_abs_path

load_dotenv()

class MaterialsProjectDownloader:
    """封装 Materials Project 材料下载、去重与记录逻辑。"""

    def __init__(self):
        self.api_key = os.environ.get("MP_API_KEY", "")
        self.workspace_dir = get_abs_path(mp_config["download_dir"])
        self.time_str = datetime.now().strftime("%m")
        self.history_file = get_abs_path(mp_config["history_file"])
        os.makedirs(self.workspace_dir, exist_ok=True)

    # @staticmethod
    # def _sanitize_filename(name: str) -> str:
    #     """将字符串中的非法文件名字符替换为下划线。"""
    #     return re.sub(r'[\\/*?:"<>|]', "_", name).strip()

    def get_downloaded_formulas(self) -> set:
        """
        获取所有已下载材料的化学式集合。

        Returns:
            已下载化学式的集合，如 {"LiFePO4", "ZnO"}。
        """
        if not os.path.exists(self.history_file):
            return set()
        formulas = set()
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    formulas.add(line.split(" (")[0].strip())
        return formulas

    def is_duplicate(self, formula: str) -> bool:
        """
        判断指定化学式的材料是否已存在于本地下载历史中。

        Args:
            formula: 材料化学式，如 "ZnO"。

        Returns:
            True 表示已下载过，False 表示未下载。
        """
        return formula in self.get_downloaded_formulas()

    def record_download(self, formula: str, mp_id: str) -> None:
        """
        将成功下载的材料记录到历史文件，防止后续重复下载。

        Args:
            formula: 材料化学式。
            mp_id: Materials Project 材料 ID。
        """
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(f"{formula} ({mp_id})\n")

    def query_material_properties(self, formula: str) -> dict | None:
        """
        通过化学式从 Materials Project 查询材料信息。
        自动检索最稳定的结构并提取关键物理属性。

        Args:
            formula: 材料化学式，如 "ZnO"。

        Returns:
            包含材料属性的字典（含 Structure 对象）；若未找到或查询失败则返回 None。
        """
        if not self.api_key:
            return None

        try:
            mpr = MPRester(self.api_key)

            # 1. 一次性批量获取所有匹配材料的摘要，按形成能排序（避免 N+1 查询）
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
            summary_docs = list(mpr.materials.summary.search(
                formula=formula, fields=summary_fields
            ))
            if not summary_docs:
                return None

            # 按形成能从低到高排序（最稳定在前）
            def _formation_energy(s):
                return s.formation_energy_per_atom if s.formation_energy_per_atom is not None else float("inf")
            summary_docs.sort(key=_formation_energy)
            best_summary = summary_docs[0]

            # 2. 仅获取最稳定材料的结构（避免拉取全部 300+ 结构数据）
            structure_docs = mpr.materials.search(
                material_ids=[best_summary.material_id],
                fields=["material_id", "structure"],
            )
            if not structure_docs:
                return None

            best_doc = structure_docs[0]
            mp_id = best_doc.material_id
            structure = best_doc.structure
            best_energy = best_summary.formation_energy_per_atom

            # 3. 提取晶体结构信息
            sga = SpacegroupAnalyzer(structure)
            conv_struct = sga.get_conventional_standard_structure()
            lattice = conv_struct.lattice
            a, b, c = [round(x, 3) for x in lattice.abc]
            alpha, beta, gamma = [round(x, 3) for x in lattice.angles]

            # 4. 组装属性字典
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

            return {
                "Formula": formula,
                "MP_ID": mp_id,
                "Structure": structure,
                "Number of Atoms": conv_struct.num_sites,
                "Volume (Å³)": round(lattice.volume, 3),
                "Lattice a,b,c (Å)": f"{a}, {b}, {c}",
                "Angles α,β,γ (°)": f"{alpha}, {beta}, {gamma}",
                "Crystal System": best_summary.symmetry.crystal_system.value
                if best_summary.symmetry
                else "未知",
                "Space Group": best_summary.symmetry.symbol
                if best_summary.symmetry
                else "未知",
                "Density (g/cm³)": round(best_summary.density, 3)
                if best_summary.density is not None
                else "未知",
                "Band Gap (eV)": round(best_summary.band_gap, 3)
                if best_summary.band_gap is not None
                else "未知",
                "Is Magnetic": best_summary.is_magnetic,
                "Formation Energy (eV/atom)": round(best_energy, 3)
                if best_energy != float("inf")
                else "未知",
                "Energy Above Hull (eV/atom)": e_hull,
                "Bulk Modulus (GPa)": k_modulus,
            }

        except Exception as e:
            logger.error(f"查询 {formula} 属性时发生异常: {e}")
            return None

    def _save_files(self, formula: str, properties: dict) -> str:
        """
        将材料结构和属性文件保存到本地目录（按化学式分子创建独立子目录）。

        Returns:
            保存目录的绝对路径。
        """
        structure = properties.pop("Structure")
        save_dir = os.path.join(self.workspace_dir, f"dataset{self.time_str}", formula)
        os.makedirs(save_dir, exist_ok=True)

        poscar_path = os.path.join(save_dir, "POSCAR")
        sga = SpacegroupAnalyzer(structure)
        conv_struct = sga.get_conventional_standard_structure()
        conv_struct.to(fmt="poscar", filename=poscar_path)

        json_path = os.path.join(save_dir, "properties.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(properties, f, ensure_ascii=False, indent=4)

        return os.path.abspath(save_dir)

    def download(self, formula: str) -> str:
        """
        从 Materials Project 下载指定化学式的最稳定材料结构及属性。

        Args:
            formula: 目标材料化学式，如 "ZnO"。

        Returns:
            包含下载结果或错误提示的字符串。
        """
        if not self.api_key:
            return "❌ 未配置 Materials Project API Key，请在 agent_config.yaml 中设置 mp_api_key 或配置环境变量 MP_API_KEY。"

        if self.is_duplicate(formula):
            return f"⚠️ 材料 {formula} 已存在于本地历史记录中，跳过下载。"

        properties = self.query_material_properties(formula)
        if properties is None:
            return f"❌ 在 Materials Project 中未找到化学式为 {formula} 的材料，或查询过程中发生异常。"

        mp_id = properties["MP_ID"]
        save_dir = self._save_files(formula, properties)
        self.record_download(formula, mp_id)

        logger.info(
            f"成功从 Materials Project 下载 {formula} (MP_ID: {mp_id})，保存至 {save_dir}"
        )
        return f"✅ 成功下载 {formula} (MP_ID: {mp_id})。文件已保存在: {save_dir}"

    def list_history(self) -> str:
        """
        列出本地已下载的所有 Materials Project 材料历史记录。

        Returns:
            格式化的历史记录字符串。
        """
        if not os.path.exists(self.history_file):
            return "📭 本地暂无已下载的材料记录。"

        with open(self.history_file, "r", encoding="utf-8") as f:
            history = [line.strip() for line in f if line.strip()]

        if not history:
            return "📭 本地暂无已下载的材料记录。"

        return "📋 已下载材料列表（共 {} 条）：\n".format(len(history)) + "\n".join(
            history
        )
