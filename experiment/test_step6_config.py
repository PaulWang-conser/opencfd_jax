"""第六步的配置、放行校验与入口（U5）。

锁六件事：
  1. 放行的多块配置返回拓扑并打印接口报告；
  2. 没验收过的开关都在读坐标之前被拦住：生效值（粘性导数的阶数、精度、格式、步长、
     算例、特殊边界脚本）和显式给出的参数（``--weno-tv``、``--outlet-ghost``、网格尺寸、
     内置 plate 专用的几个参数）各查各的；粘性本身从第七步起放行，只放行 CD8；
  3. 参考尺度必须给出、是有限正数、不小于任何一块按自身尺寸算出的分量；没有走
     多块路径时给出了也报错；
  4. 没有多 zone 的时候走原路径，报错与改动前相同；显式给了 ``--cgns-zone ""`` 算
     显式选了 zone；
  5. 文件残缺时不漏裸异常：入口探测只为选路径，探不出来就交回原路径；多 zone 文件
     里有读不出来的 zone、坐标写成组或文字、Base 下有打不开的链接，都报成“多块网格
     没有通过校验”；
  6. 命令行入口：不放行的运行只输出一行“初始化错误”，退出码 2；帮助文本写明
     不带 ``--cgns-zone`` 时读全部 zone。
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

import jax.numpy as jnp  # noqa: E402

from opencfd_jax.src import init, main, mesh_cgns, multiblock, runtime_config  # noqa: E402
from opencfd_jax.src.errors import SetupError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402
from Grid.make_cgns_fixture import _group  # noqa: E402

PLATE = support.SOURCE
POISEUILLE = REPO / "Grid" / "fixtures" / "poiseuille_box.cgns"
pytestmark = pytest.mark.skipif(not PLATE.exists(), reason="缺少源网格")

FLOW = {"rho": 1.0, "u": 1.0, "v": 0.0, "w": 0.0, "T": 1.0}
ConfigError = runtime_config.ConfigError


def _cli(*argv):
    """按 main.main 的方式把命令行解析成“用户显式给过的参数”。"""
    parser = main.build_parser(explicit_only=True)
    _, explicit = main.resolve_cli(vars(parser.parse_args(argv)), parser)
    return explicit


def _config(mesh, overrides=None):
    """一份放行的多块 plate 配置；``overrides`` 按点路径改，值是 None 表示删掉。"""
    cfg = {
        "case": {"name": "plate"},
        "mesh": {"format": "cgns", "file": str(mesh)},
        "boundary": {"inflow": dict(FLOW), "farfield": dict(FLOW),
                     "wall_temperature": 1.0},
        "physics": {"Re": 1000.0, "Ama": 0.1, "ref_t": 288.15, "pr": 0.7},
        "numerics": {"scheme": "WENO7", "vis_order": "CD8", "viscous": False,
                     "flux_scale": [1521, 3861, 3861]},
        "time": {"dt": 0.0001, "end_time": 0.0002, "end_step": 2},
        "output": {"dir": "out", "kstep_show": 1, "kstep_save": 1},
    }
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".", 1)
        if len(parts) == 1:                    # 顶层的单值字段（precision 等）
            if value is None:
                cfg.pop(parts[0], None)
            else:
                cfg[parts[0]] = value
            continue
        section, name = parts
        if value is None:
            cfg.setdefault(section, {}).pop(name, None)
        else:
            cfg.setdefault(section, {})[name] = value
    return cfg


def _resolve(tmp_path, overrides=None, argv=(), mesh=None):
    """解析出 args：JSON 放在临时目录，再叠加命令行。"""
    path = tmp_path / "case.json"
    path.write_text(json.dumps(_config(mesh or support.mesh_path("x2"),
                                       overrides)), encoding="utf-8")
    return runtime_config.resolve_args(
        _cli(*argv), runtime_config.load_config(str(path)),
        main.build_parser().parse_args([]))


def _broken_mesh(tmp_path):
    """x2 的副本，第二个 zone 的 CoordinateX 被删掉：读元数据没问题，读坐标会报错。"""
    path = tmp_path / "broken.cgns"
    shutil.copyfile(support.mesh_path("x2"), path)
    with h5py.File(path, "r+") as f:
        del f["Base/blk-i1j0k0/GridCoordinates/CoordinateX"]
    return path


def _second_base(path, name):
    """给一个 CGNS 文件再加一个空的 Base。"""
    with h5py.File(path, "r+") as f:
        _group(f, name, "CGNSBase_t", [3, 3], dtype=np.int32, track_times=False)


# ---- 1. 放行 ----

def test_released_multiblock_config_returns_topology_and_prints_report(tmp_path, capsys):
    setup = multiblock.enter(_resolve(tmp_path))
    assert [zone.name for zone in setup.topology.zones] == ["blk-i0j0k0", "blk-i1j0k0"]
    assert setup.flux_scale == (1521.0, 3861.0, 3861.0)
    assert "接口对 1 个（普通切口 1 个" in capsys.readouterr().out


def test_command_line_can_switch_off_a_viscous_json(tmp_path):
    setup = multiblock.enter(_resolve(tmp_path, {"numerics.viscous": True},
                                      ("--no-viscous",)))
    assert len(setup.topology.zones) == 2


# ---- 2. 不放行的配置，都拦在读坐标之前 ----

UNRELEASED = [
    # (JSON 的改动, 命令行, 报错里应当出现的字样)
    # 粘性可以打开（第七步），但粘性导数只放行 CD8：JSON 给的 CD6、命令行给的 CD6 都拦
    ({"numerics.viscous": True, "numerics.vis_order": "CD6"}, (), "vis_order"),
    ({}, ("--viscous", "--vis-order", "CD6"), "vis_order"),
    ({"precision": "fp32"}, (), "FP64"),
    ({}, ("--fp32",), "FP64"),
    ({"numerics.scheme": "UP7"}, (), "WENO7"),
    ({}, ("--scheme", "UP7"), "WENO7"),
    ({"numerics.weno_tv": True}, (), "weno-tv"),
    ({}, ("--weno-tv",), "weno-tv"),
    ({}, ("--no-weno-tv",), "weno-tv"),
    ({"boundary.outlet_ghost": "keep"}, (), "outlet-ghost"),
    ({}, ("--outlet-ghost", "keep"), "outlet-ghost"),
    ({}, ("--ni", "100"), "--ni"),            # 与网格相同的尺寸，显式给了也算
    ({}, ("--nj", "40"), "--nj"),
    ({}, ("--nk", "40"), "--nk"),
    ({"case.name": "poiseuille", "boundary.inflow": None,
      "boundary.farfield": None, "boundary.wall_temperature": None}, (), "plate"),
    ({"time.dt": None}, (), "dt"),
    ({}, ("--dt", "nan"), "dt"),
    ({"specificboundary": "trip.py"}, (), "specificboundary"),
    ({}, ("--tw", "1.0"), "--tw"),
    ({}, ("--inflow-profile", "unused.txt"), "--inflow-profile"),
    ({}, ("--plate-x0", "0.1"), "--plate-x0"),
    ({}, ("--plate-x1", "0.25"), "--plate-x1"),
    ({}, ("--plate-h", "0.12"), "--plate-h"),
    ({}, ("--plate-dif-bound", "0 0 0 0 1 0"), "--plate-dif-bound"),
    ({}, ("--kstep-show", "0"), "kstep_show"),
    ({}, ("--kstep-save", "-1"), "kstep_save"),
    # 单块路径要求壁温大于 0；多块路径不拦的话会按绝热壁算下去，面上却标着等温壁
    ({"boundary.wall_temperature": 0.0}, (), "wall_temperature 大于 0"),
    ({"boundary.wall_temperature": -1.0}, (), "wall_temperature 大于 0"),
]


@pytest.mark.parametrize("overrides, argv, text", UNRELEASED,
                         ids=[f"json{o} cli{a}" for o, a, _ in UNRELEASED])
def test_unreleased_settings_are_rejected_before_coordinates_are_read(
        tmp_path, overrides, argv, text):
    broken = _broken_mesh(tmp_path)
    with pytest.raises((SetupError, ConfigError)) as info:
        multiblock.enter(_resolve(tmp_path, overrides, argv, mesh=broken))
    assert text in str(info.value), str(info.value)


def test_reader_errors_surface_once_the_config_is_released(tmp_path):
    with pytest.raises(SetupError, match="CoordinateX"):
        multiblock.enter(_resolve(tmp_path, mesh=_broken_mesh(tmp_path)))


def test_pure_command_line_without_boundary_states_is_rejected(tmp_path):
    explicit = _cli("--case", "plate", "--mesh", str(support.mesh_path("x2")),
                    "--dt", "1e-4", "--scheme", "WENO7", "--no-viscous",
                    "--flux-scale", "1521", "3861", "3861")
    args = runtime_config.resolve_args(explicit, None, main.build_parser().parse_args([]))
    with pytest.raises(ConfigError, match=r"boundary\.(inflow|farfield|wall_temperature)"):
        multiblock.enter(args)


# ---- 3. 参考尺度 ----

def test_flux_scale_must_be_given_and_finitely_positive(tmp_path):
    with pytest.raises(ConfigError, match="flux_scale"):
        multiblock.enter(_resolve(tmp_path, {"numerics.flux_scale": None}))
    with pytest.raises(ConfigError, match=r"flux_scale\[1\]"):
        _resolve(tmp_path, {"numerics.flux_scale": [1521, 0, 3861]})
    with pytest.raises(ConfigError, match="flux_scale"):
        multiblock.enter(_resolve(tmp_path, argv=("--flux-scale", "1521", "nan", "3861")))


def test_flux_scale_filled_from_one_block_is_rejected(tmp_path):
    args = _resolve(tmp_path, {"numerics.flux_scale": [361, 931, 931]},
                    mesh=support.mesh_path("xyz8"))     # 按 50×20×20 的那一块填的
    with pytest.raises(ConfigError) as info:
        multiblock.enter(args)
    message = str(info.value)
    assert "blk-i0j0k0" in message and "361" in message and "400" in message


def test_flux_scale_outside_the_multiblock_path_is_rejected(tmp_path):
    explicit = _cli("--case", "plate", "--dt", "1e-4",
                    "--flux-scale", "1521", "3861", "3861")
    with pytest.raises(ConfigError, match="flux_scale"):        # 内置网格的 plate
        multiblock.enter(runtime_config.resolve_args(
            explicit, None, main.build_parser().parse_args([])))
    with pytest.raises(ConfigError, match="flux_scale"):        # 单 zone 的文件
        multiblock.enter(_resolve(tmp_path, mesh=PLATE))
    with pytest.raises(ConfigError, match="flux_scale"):        # 显式选了 zone
        multiblock.enter(_resolve(tmp_path, argv=("--cgns-zone", "blk-i0j0k0")))


# ---- 4. 没有多 zone 时走原路径 ----

def test_unreadable_single_zone_mesh_keeps_its_old_error(tmp_path):
    bad = tmp_path / "bad.cgns"
    bad.write_text("not an HDF5 file", encoding="utf-8")
    args = _resolve(tmp_path, {"numerics.flux_scale": None}, mesh=bad)
    assert multiblock.enter(args) is None                    # 交给原路径
    with pytest.raises(SetupError, match="无法打开") as info:
        main.run(args)
    assert str(info.value).startswith("初始化失败：case=plate")


def test_zone_count_decides_the_path_when_the_file_has_two_bases(tmp_path):
    one = tmp_path / "one_zone_two_bases.cgns"
    shutil.copyfile(POISEUILLE, one)
    _second_base(one, "Base 2")
    assert multiblock.enter(_resolve(tmp_path, {"numerics.flux_scale": None}, mesh=one)) is None
    assert mesh_cgns.read_cgns_zone(one).name == mesh_cgns.read_cgns_zone(POISEUILLE).name

    many = tmp_path / "two_zones_two_bases.cgns"
    shutil.copyfile(support.mesh_path("x2"), many)
    _second_base(many, "Base 2")
    with pytest.raises(SetupError, match="CGNSBase_t"):
        multiblock.enter(_resolve(tmp_path, mesh=many))


def _second_zone_data(path, data, zone_type=None):
    """把 x2 副本的第二个 zone 的尺寸数据换成 ``data``，需要时连 ``ZoneType`` 一起换。"""
    with h5py.File(path, "r+") as f:
        node = f["Base/blk-i1j0k0"]
        del node[" data"]
        node.create_dataset(" data", data=data)
        if zone_type is not None:
            del node["ZoneType"][" data"]
            node["ZoneType"].create_dataset(
                " data", data=np.frombuffer(zone_type.encode(), dtype=np.int8))


def test_unreadable_zone_in_a_multizone_file_is_reported_by_the_multiblock_path(tmp_path):
    """文件里不止一个 zone、其中有读不出来的：由多块路径报错，不能悄悄退回单块路径。"""
    mixed = tmp_path / "unstructured.cgns"
    shutil.copyfile(support.mesh_path("x2"), mixed)
    _second_zone_data(mixed, np.array([[7], [6], [0]], dtype=np.int32), "Unstructured")
    with pytest.raises(SetupError, match="多块网格没有通过校验") as info:
        multiblock.enter(_resolve(tmp_path, mesh=mixed))
    assert "blk-i1j0k0" in str(info.value) and "Unstructured" in str(info.value)
    unreadable = tmp_path / "text_size.cgns"
    shutil.copyfile(support.mesh_path("x2"), unreadable)
    _second_zone_data(unreadable, np.full((3, 3), b"x", dtype="S1"))
    with pytest.raises(SetupError, match="多块网格没有通过校验") as info:
        multiblock.enter(_resolve(tmp_path, mesh=unreadable))
    assert "blk-i1j0k0" in str(info.value)


def test_undecodable_size_in_a_single_zone_file_keeps_the_old_error(tmp_path):
    bad = tmp_path / "text_size.cgns"
    shutil.copyfile(PLATE, bad)
    with h5py.File(bad, "r+") as f:
        node = next(obj for obj in f["Base"].values() if "ZoneType" in obj)
        del node[" data"]
        node.create_dataset(" data", data=np.full((3, 3), b"x", dtype="S1"))
    args = _resolve(tmp_path, {"numerics.flux_scale": None}, mesh=bad)
    assert multiblock.enter(args) is None                 # 交给原路径
    with pytest.raises(SetupError, match="初始化失败：case=plate") as info:
        main.run(args)
    assert "invalid literal" in str(info.value)


def test_empty_cgns_zone_keeps_the_single_block_path(tmp_path):
    """显式给了 ``--cgns-zone ""`` 就是显式选了 zone，和没给不是一回事。"""
    args = _resolve(tmp_path, {"numerics.flux_scale": None}, argv=("--cgns-zone", ""))
    assert multiblock.enter(args) is None
    with pytest.raises(SetupError, match="初始化失败") as info:
        main.run(args)
    assert "找不到唯一 zone ''" in str(info.value)


def test_damaged_zone_type_node_is_reported_without_a_raw_exception(tmp_path):
    """``ZoneType`` 的数据集被删：多 zone 文件按多块校验报错并点名 zone；单 zone
    文件交回原路径，照旧先报尺寸的问题（与没有多块路径时相同）。"""
    many = tmp_path / "no_zone_type.cgns"
    shutil.copyfile(support.mesh_path("x2"), many)
    with h5py.File(many, "r+") as f:
        del f["Base/blk-i1j0k0/ZoneType"][" data"]
    with pytest.raises(SetupError, match="多块网格没有通过校验") as info:
        multiblock.enter(_resolve(tmp_path, mesh=many))
    assert "blk-i1j0k0" in str(info.value)
    one = tmp_path / "one_no_zone_type.cgns"
    shutil.copyfile(PLATE, one)
    with h5py.File(one, "r+") as f:
        node = next(obj for obj in f["Base"].values() if "ZoneType" in obj)
        del node["ZoneType"][" data"]
        del node[" data"]
        node.create_dataset(" data", data=np.array([[7], [6], [0]], dtype=np.int32))
    args = _resolve(tmp_path, {"numerics.flux_scale": None}, mesh=one)
    assert multiblock.enter(args) is None
    with pytest.raises(SetupError, match="初始化失败：case=plate") as info:
        main.run(args)
    assert "Zone_t data 需要 (3,3)" in str(info.value)


def test_damaged_mesh_files_never_raise_raw_exceptions(tmp_path):
    """文件残缺的方式很多（Astra 第 18、19 轮各找到几种）。读文件这一层的规则是：
    入口探测只为选路径，探不出来就交回单块路径，由它照旧报错；多块路径上读取和校验
    出的任何异常都算“多块网格没有通过校验”。多块路径不许漏出裸异常；单块路径保持
    它自己的报错，残缺的文件可能在那里直接抛出原始异常，与没有多块路径时相同。"""
    # 单 zone：尺寸数据引用了不存在的外部文件。单块路径自己报什么随 h5py 的版本而变
    # （有的抛 OSError 被包成 SetupError，有的抛 KeyError），所以拿它直接运行时的报错当基准
    one = tmp_path / "external.cgns"
    shutil.copyfile(PLATE, one)
    with h5py.File(one, "r+") as f:
        node = next(obj for obj in f["Base"].values() if "ZoneType" in obj)
        where = node[" data"].name
        del node[" data"]
        f[where] = h5py.ExternalLink(str(tmp_path / "missing.h5"), "/sizes")
    args = _resolve(tmp_path, {"numerics.flux_scale": None}, mesh=one)
    assert multiblock.enter(args) is None
    with pytest.raises(Exception) as direct:
        init.prepare(args)                       # 没有多块路径时的初始化
    with pytest.raises(type(direct.value)):
        main.run(args)
    # 多 zone：坐标写成了组；坐标是文字
    for damage in ("group", "text"):
        many = tmp_path / f"coords_{damage}.cgns"
        shutil.copyfile(support.mesh_path("x2"), many)
        with h5py.File(many, "r+") as f:
            coords = f["Base/blk-i1j0k0/GridCoordinates"]
            del coords["CoordinateX"]
            if damage == "group":
                coords.create_group("CoordinateX")
            else:
                coords.create_dataset("CoordinateX", data=np.full((7, 40, 50), b"abc", "S3"))
        with pytest.raises(SetupError, match="多块网格没有通过校验") as info:
            multiblock.enter(_resolve(tmp_path, mesh=many))
        assert "blk-i1j0k0" in str(info.value), str(info.value)   # 报错里带着 zone 名
    # Base 下有悬空链接：探测数得出两个 zone，读取器点名报错。不跳过它——一个 zone
    # 自己就是打不开的外部链接时，跳过等于悄悄少算一块。
    linked = tmp_path / "dangling.cgns"
    shutil.copyfile(support.mesh_path("x2"), linked)
    with h5py.File(linked, "r+") as f:
        f["Base/nowhere"] = h5py.SoftLink("/no/such/node")
    assert len(mesh_cgns.zone_metadata(linked)) == 2
    with pytest.raises(SetupError, match="多块网格没有通过校验") as info:
        multiblock.enter(_resolve(tmp_path, mesh=linked))
    assert "'nowhere'" in str(info.value)


def test_multiblock_path_runs_in_fp64(tmp_path):
    config.disable_x64()
    try:
        multiblock.enter(_resolve(tmp_path))
        assert config.precision_name() == "float64"
        assert jnp.zeros(1).dtype == jnp.float64
    finally:
        config.enable_x64()


# ---- 5. 命令行入口 ----

def test_unreleased_run_from_the_command_line_reports_one_line(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(_config(support.mesh_path("x2"), {
        "numerics.viscous": True, "numerics.vis_order": "CD6"})), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--config", str(path)],
        cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    reported = [line for line in done.stderr.splitlines() if "初始化错误" in line]
    assert len(reported) == 1 and "vis_order" in reported[0]


def test_unreadable_mesh_from_the_command_line_keeps_one_line(tmp_path):
    bad = tmp_path / "bad.cgns"
    bad.write_text("not an HDF5 file", encoding="utf-8")
    path = tmp_path / "case.json"
    path.write_text(json.dumps(_config(bad, {"numerics.flux_scale": None})),
                    encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "opencfd_jax.src.main", "--config", str(path)],
        cwd=PROJECT, capture_output=True, text=True)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
    reported = [line for line in done.stderr.splitlines() if "初始化错误" in line]
    assert len(reported) == 1 and "无法打开" in reported[0]


def test_help_says_all_zones_are_read_without_cgns_zone():
    help_text = main.build_parser().format_help()
    assert "全部 zone" in help_text and "--flux-scale" in help_text
