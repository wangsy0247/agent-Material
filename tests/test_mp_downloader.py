"""测试 MaterialsProjectDownloader 核心逻辑。"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

import utils.config_handler as config_module
from agent.tools.mp_downloader import MaterialsProjectDownloader


class TestHistoryOperations:
    """测试下载历史相关的操作方法。"""

    @pytest.fixture
    def downloader(self, tmp_path, monkeypatch):
        """提供一个使用临时目录的下载器实例。"""
        # Mock mp_config，使下载器使用临时目录
        monkeypatch.setitem(
            config_module.mp_config, "download_dir", str(tmp_path / "downloads")
        )
        monkeypatch.setitem(
            config_module.mp_config, "history_file", str(tmp_path / "history.txt")
        )
        monkeypatch.setenv("MP_API_KEY", "test-api-key")

        d = MaterialsProjectDownloader()
        return d

    def test_get_downloaded_formulas_empty_history(self, downloader):
        """历史文件不存在时，应返回空集合。"""
        formulas = downloader.get_downloaded_formulas()
        assert formulas == set()
        assert isinstance(formulas, set)

    def test_record_and_get_formulas(self, downloader):
        """记录下载后，应能从历史中正确读取。"""
        downloader.record_download("LiFePO4", "mp-19017")
        downloader.record_download("ZnO", "mp-2133")

        formulas = downloader.get_downloaded_formulas()
        assert formulas == {"LiFePO4", "ZnO"}

    def test_is_duplicate_true(self, downloader):
        """已记录的材料应被检测为重复。"""
        downloader.record_download("TiO2", "mp-2657")
        assert downloader.is_duplicate("TiO2") is True

    def test_is_duplicate_false(self, downloader):
        """未记录的材料不应被检测为重复。"""
        downloader.record_download("TiO2", "mp-2657")
        assert downloader.is_duplicate("ZnO") is False

    def test_record_append_not_overwrite(self, downloader):
        """多次记录应追加而非覆盖。"""
        downloader.record_download("A", "mp-1")
        downloader.record_download("B", "mp-2")
        downloader.record_download("C", "mp-3")

        formulas = downloader.get_downloaded_formulas()
        assert len(formulas) == 3
        assert formulas == {"A", "B", "C"}

    def test_history_file_format(self, downloader):
        """验证历史文件的实际存储格式。"""
        downloader.record_download("Fe2O3", "mp-1234")

        with open(downloader.history_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Fe2O3 (mp-1234)" in content


class TestDownloadWithMock:
    """使用 Mock 测试需要调用 Materials Project API 的下载流程。"""

    @pytest.fixture
    def downloader(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            config_module.mp_config, "download_dir", str(tmp_path / "downloads")
        )
        monkeypatch.setitem(
            config_module.mp_config, "history_file", str(tmp_path / "history.txt")
        )
        monkeypatch.setenv("MP_API_KEY", "test-api-key")
        return MaterialsProjectDownloader()

    def test_download_duplicate_skipped(self, downloader):
        """重复材料应直接返回跳过提示，不调用 API。"""
        downloader.record_download("ZnO", "mp-2133")

        result = downloader.download("ZnO")
        assert "已存在于本地历史记录中" in result
        assert "跳过下载" in result

    def test_download_without_api_key(self, tmp_path, monkeypatch):
        """未配置 API Key 时应返回配置提示。"""
        monkeypatch.setitem(
            config_module.mp_config, "download_dir", str(tmp_path / "downloads")
        )
        monkeypatch.setitem(
            config_module.mp_config, "history_file", str(tmp_path / "history.txt")
        )
        monkeypatch.delenv("MP_API_KEY", raising=False)
        # 同时清理 agent_config 中的 key
        original_key = config_module.agent_config.get("mp_api_key", "")
        monkeypatch.setitem(config_module.agent_config, "mp_api_key", "")

        d = MaterialsProjectDownloader()
        result = d.download("ZnO")
        assert "未配置 Materials Project API Key" in result

    @patch("agent.tools.mp_downloader.MPRester")
    def test_download_success(self, mock_mpr_class, downloader, tmp_path):
        """模拟 MP API 返回有效数据，验证下载成功流程。"""
        mock_mpr = MagicMock()
        mock_mpr_class.return_value = mock_mpr

        # Mock materials.search 返回一个材料文档
        mock_doc = MagicMock()
        mock_doc.material_id = "mp-test-001"
        mock_structure = MagicMock()
        mock_doc.structure = mock_structure
        mock_mpr.materials.search.return_value = [mock_doc]

        # Mock materials.summary.search 返回摘要
        mock_summary = MagicMock()
        mock_summary.formation_energy_per_atom = -1.5
        mock_summary.energy_above_hull = 0.02
        mock_summary.band_gap = 3.2
        mock_summary.density = 5.6
        mock_summary.is_magnetic = False
        mock_summary.bulk_modulus = {"vrh": 150}

        mock_sym = MagicMock()
        mock_sym.crystal_system.value = "cubic"
        mock_sym.symbol = "Fd-3m"
        mock_summary.symmetry = mock_sym

        mock_mpr.materials.summary.search.return_value = [mock_summary]

        # Mock SpacegroupAnalyzer
        with patch("agent.tools.mp_downloader.SpacegroupAnalyzer") as mock_sga_class:
            mock_sga = MagicMock()
            mock_conv_struct = MagicMock()
            mock_conv_struct.num_sites = 4
            mock_lattice = MagicMock()
            mock_lattice.abc = (4.0, 4.0, 4.0)
            mock_lattice.angles = (90.0, 90.0, 90.0)
            mock_lattice.volume = 64.0
            mock_conv_struct.lattice = mock_lattice

            # 让 mock 的 to 方法实际创建一个空文件，模拟 POSCAR 输出
            def _fake_to(fmt, filename):
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("FAKE POSCAR CONTENT\n")

            mock_conv_struct.to.side_effect = _fake_to
            mock_sga.get_conventional_standard_structure.return_value = mock_conv_struct
            mock_sga_class.return_value = mock_sga

            result = downloader.download("ZnO")

        assert "成功下载" in result
        assert "ZnO" in result
        assert "mp-test-001" in result

        # 验证文件已保存到按化学式命名的子目录
        dataset_dir = tmp_path / "downloads" / f"dataset{downloader.time_str}" / "ZnO"
        assert dataset_dir.exists()
        assert (dataset_dir / "POSCAR").exists()
        assert (dataset_dir / "properties.json").exists()

        # 验证 properties.json 内容
        with open(dataset_dir / "properties.json", "r", encoding="utf-8") as f:
            props = json.load(f)
        assert props["Formula"] == "ZnO"
        assert props["MP_ID"] == "mp-test-001"
        assert props["Band Gap (eV)"] == 3.2

    @patch("agent.tools.mp_downloader.MPRester")
    def test_download_material_not_found(self, mock_mpr_class, downloader):
        """MP 中不存在该材料时，应返回未找到提示。"""
        mock_mpr = MagicMock()
        mock_mpr_class.return_value = mock_mpr
        mock_mpr.materials.search.return_value = []

        result = downloader.download("FakeMaterial123")
        assert "未找到" in result

    @patch("agent.tools.mp_downloader.MPRester")
    def test_download_api_exception(self, mock_mpr_class, downloader):
        """MP API 异常时，应返回错误提示（异常被 query_material_properties 捕获后返回 None，download 给出通用错误）。"""
        mock_mpr_class.side_effect = Exception("Network timeout")

        result = downloader.download("ZnO")
        # 异常发生在 query_material_properties 内部，被捕获后返回 None
        # download 方法看到 None 后返回"未找到...或查询过程中发生异常"
        assert "未找到" in result or "发生异常" in result


class TestQueryMaterialProperties:
    """测试 query_material_properties 查询方法。"""

    @pytest.fixture
    def downloader(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            config_module.mp_config, "download_dir", str(tmp_path / "downloads")
        )
        monkeypatch.setitem(
            config_module.mp_config, "history_file", str(tmp_path / "history.txt")
        )
        monkeypatch.setenv("MP_API_KEY", "test-api-key")
        return MaterialsProjectDownloader()

    def test_query_without_api_key(self, tmp_path, monkeypatch):
        """未配置 API Key 时应返回 None。"""
        monkeypatch.setitem(
            config_module.mp_config, "download_dir", str(tmp_path / "downloads")
        )
        monkeypatch.setitem(
            config_module.mp_config, "history_file", str(tmp_path / "history.txt")
        )
        monkeypatch.delenv("MP_API_KEY", raising=False)
        monkeypatch.setitem(config_module.agent_config, "mp_api_key", "")

        d = MaterialsProjectDownloader()
        result = d.query_material_properties("ZnO")
        assert result is None

    @patch("agent.tools.mp_downloader.MPRester")
    def test_query_success_returns_dict(self, mock_mpr_class, downloader):
        """查询成功应返回包含正确字段的字典。"""
        mock_mpr = MagicMock()
        mock_mpr_class.return_value = mock_mpr

        mock_doc = MagicMock()
        mock_doc.material_id = "mp-query-001"
        mock_doc.structure = MagicMock()
        mock_mpr.materials.search.return_value = [mock_doc]

        mock_summary = MagicMock()
        mock_summary.formation_energy_per_atom = -2.0
        mock_summary.energy_above_hull = 0.0
        mock_summary.band_gap = 1.5
        mock_summary.density = 4.0
        mock_summary.is_magnetic = True
        mock_summary.bulk_modulus = None

        mock_sym = MagicMock()
        mock_sym.crystal_system.value = "hexagonal"
        mock_sym.symbol = "P63/mmc"
        mock_summary.symmetry = mock_sym

        mock_mpr.materials.summary.search.return_value = [mock_summary]

        with patch("agent.tools.mp_downloader.SpacegroupAnalyzer") as mock_sga_class:
            mock_sga = MagicMock()
            mock_conv = MagicMock()
            mock_conv.num_sites = 2
            mock_lat = MagicMock()
            mock_lat.abc = (3.0, 3.0, 5.0)
            mock_lat.angles = (90.0, 90.0, 120.0)
            mock_lat.volume = 38.97
            mock_conv.lattice = mock_lat
            mock_sga.get_conventional_standard_structure.return_value = mock_conv
            mock_sga_class.return_value = mock_sga

            result = downloader.query_material_properties("TiO2")

        assert result is not None
        assert isinstance(result, dict)
        assert result["Formula"] == "TiO2"
        assert result["MP_ID"] == "mp-query-001"
        assert result["Band Gap (eV)"] == 1.5
        assert result["Is Magnetic"] is True
        assert result["Bulk Modulus (GPa)"] == "未知"
        assert "Structure" in result

    @patch("agent.tools.mp_downloader.MPRester")
    def test_query_material_not_found(self, mock_mpr_class, downloader):
        """材料不存在时应返回 None。"""
        mock_mpr = MagicMock()
        mock_mpr_class.return_value = mock_mpr
        mock_mpr.materials.search.return_value = []

        result = downloader.query_material_properties("NonExistentMaterial")
        assert result is None
