"""第 0 步（固定当前基准）用例清单。

每个用例就是一条**真实入口命令** ``python -m opencfd_jax.src.run <argv>``，
不复制 run.py 的解析逻辑，避免以后改 run.py 时基准与实现脱钩。

统一口径（用户 2026-11-17 指定）：
  - 无粘：``WENO7``（原码 SCHEME_INVIS=303 的 7 阶 WENO-Z），**不用 OMP6 系列**；
  - 粘性：``--vis-order CD8``；
  - FP64；激波滤波关闭（``--filter`` 不启用），保证"格式本身"被冻结；
  - 固定 dt，保证逐点可复现（另设一条 CFL 自适应用例单独冻结 compute_dt）。

第 0 步只做一件事：把这些结果连同哈希一起钉住，后续第 1~9 步每一步都要
复跑本清单并给出 ``check_step0.py`` 的 PASS。
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]                     # ...\ocfd2jax
QUICK = PROJECT / "channel_case_iiA" / "quick"
SMALL_IN = HERE / "cases" / "channel_small.in"

CASES = [
    {
        "id": "sod_1step",
        "title": "Sod 激波管 · 1 步（无粘，固定 dt）",
        "cover": "无粘通量分裂 + WENO7 重构 + 出流/周期边界；单步 du 的冻结",
        "argv": ["--case", "sod", "--N", "128", "--ny", "4", "--nz", "4",
                 "--scheme", "WENO7", "--vis-order", "CD8",
                 "--dt", "1e-3", "--end-step", "1", "--end-time", "0.5",
                 "--kstep-show", "1", "--kstep-save", "1"],
        "note": "1 维问题在 3 维布局里跑（ny=nz=4）；激波尚在隔膜附近，场应保持对称。",
    },
    {
        "id": "sod_cfl_5step",
        "title": "Sod 激波管 · 5 步（CFL 自适应 dt）",
        "cover": "compute_dt 的 CFL 判据（含声速/网格/粘性口径），逐步 dt 序列被冻结",
        "argv": ["--case", "sod", "--N", "128", "--ny", "4", "--nz", "4",
                 "--scheme", "WENO7", "--vis-order", "CD8", "--cfl", "0.4",
                 "--end-step", "5", "--end-time", "0.5",
                 "--kstep-show", "1", "--kstep-save", "1"],
        "note": "每步都存 npz，_meta 里的 dt 就是自适应步长序列。",
    },
    {
        "id": "tg_10step",
        "title": "Taylor-Green 涡 · 10 步（粘性 Re=1000）",
        "cover": "周期双轴 + 粘性 CD8 + 2D 涡演化；无粘/粘性组合的单步与多步",
        "argv": ["--case", "tg", "--N", "64", "--nz", "4",
                 "--scheme", "WENO7", "--vis-order", "CD8", "--viscous",
                 "--Re", "1000", "--dt", "2e-3", "--end-step", "10",
                 "--end-time", "1.0", "--kstep-show", "5", "--kstep-save", "5"],
        "note": "Re=50 太耗散（原文档示例值），这里取 Re=1000 让耗散项真正参与。",
    },
    {
        "id": "poiseuille_5step",
        "title": "Poiseuille 层流 · 5 步（壁面 + 体积力）",
        "cover": "WALL 镜像鬼胞、体积力源项、粘性 CD8；已知壁面通量泄漏的当前表现",
        "argv": ["--case", "poiseuille", "--N", "32", "--ny", "64", "--nz", "4",
                 "--scheme", "WENO7", "--vis-order", "CD8", "--viscous",
                 "--Re", "100", "--Ama", "0.05", "--p-grad", "0.1",
                 "--dt", "1e-3", "--end-step", "5", "--end-time", "1.0",
                 "--kstep-show", "5", "--kstep-save", "5"],
        "note": ("tests/test_wall_poiseuille.py::test_poiseuille_steady 目前是 skip（壁面"
                 "邻格质量通量泄漏 → 长跑发散）。第 0 步只冻结前 5 步的**有限**演化；"
                 "等 boundary.py 的单侧壁面公式落地后，这条用例应当由 NaN/发散转为"
                 "与解析抛物线一致——届时需要显式更新基准并记录原因。"),
    },
    {
        "id": "plate_1step",
        "title": "平板层流边界层 · 1 步（入口/壁面/远场 + 曲线无关）",
        "cover": "INFLOW/PLATE 不对称边界、零状态鬼层、dif_bound 边界带、粘性 CD8",
        "argv": ["--case", "plate", "--N", "64", "--ny", "64", "--nz", "4",
                 "--scheme", "WENO7", "--vis-order", "CD8", "--viscous",
                 "--dt", "1e-4", "--end-step", "1", "--end-time", "1.0",
                 "--kstep-show", "1", "--kstep-save", "1"],
        "note": "验收基准仍是 blasius.py；这里只钉住单步 RHS 的实现细节。",
    },
    {
        "id": "channel_1step",
        "title": "单块槽道（真网格 96×97×64）· 1 步",
        "cover": "原版 in 文件 + 原版初场 + OCFD3d-Mesh.dat 度量链 + IBC=200 双壁",
        "argv": ["--case", "channel", "--in", str(QUICK / "opencfd-scu.in"),
                 "--field", str(QUICK / "opencfd.dat"),
                 "--mesh", str(QUICK / "OCFD3d-Mesh.dat"),
                 "--scheme", "WENO7", "--vis-order", "CD8", "--viscous",
                 "--forcing-work", "off",
                 "--dt", "5e-5", "--end-step", "1", "--end-time", "0.4",
                 "--kstep-show", "1", "--kstep-save", "1"],
        "note": ("覆盖 JAC_BOUND 度量（Ajac/余子式）+ 真实湍流初场 + 双壁 BC + 体积力。"
                 "输出 22.7 MB，是整套基准里最大的一份。"
                 "必须 --forcing-work off：原码 channel_body_force 只做 du[1] += F*rho，"
                 "不做 rho*u*G 的做功项，否则 rhoE 会有 ~1e-3 的系统差（实测）。"),
    },
    {
        "id": "channel_small_100step",
        "title": "单块槽道（小型 24×25×16）· 100 步",
        "cover": "多步连续演化：壁面 BC 每子步重置、体积力、粘性，累计到 t=0.005",
        "argv": ["--case", "channel", "--in", str(SMALL_IN),
                 "--scheme", "WENO7", "--vis-order", "CD8", "--viscous",
                 "--forcing-work", "off",
                 "--dt", "5e-5", "--end-step", "100", "--end-time", "0.005",
                 "--kstep-show", "20", "--kstep-save", "20"],
        "note": ("配合 channel_1step 使用：那条冻结「单步 + 真网格」，这条冻结"
                 "「多步 + 便宜网格」。现有 128×129×96 的 20000 步原版长跑结果"
                 "（channel_case_iiA/main_128x129x96）第 0 步不重跑，作为外部参考保留。"),
    },
]


def by_id(case_id):
    """按 id 取用例，找不到时报出全部可用 id。"""
    for c in CASES:
        if c["id"] == case_id:
            return c
    raise KeyError(f"unknown case id {case_id!r}; 可用: {[c['id'] for c in CASES]}")
