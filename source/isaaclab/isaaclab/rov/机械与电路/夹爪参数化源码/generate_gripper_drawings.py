from pathlib import Path
from math import cos, sin, pi
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "gripper"
OUT.mkdir(parents=True, exist_ok=True)

# All dimensions are mm. This is the single source of truth for the drawings.
P = {
    "base_w": 170.0,
    "base_h": 160.0,
    "base_t": 8.0,
    "rail_d": 8.0,
    "rail_spacing": 64.0,
    "rail_x0": 36.0,
    "rail_x1": 144.0,
    "fixed_x": 25.0,
    "moving_x_closed": 31.0,
    "moving_x_open": 151.0,
    "jaw_len": 145.0,
    "jaw_w": 24.0,
    "jaw_t": 8.0,
    "carriage_w": 48.0,
    "carriage_h": 86.0,
    "mount_hole_d": 6.5,
    "mount_x": 15.0,
    "mount_y": 15.0,
    "wrist_bcd": 50.0,
    "wrist_hole_d": 5.5,
    "wrist_center_x": 85.0,
    "wrist_center_y": 80.0,
}


def svg_header(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
        f'viewBox="0 0 {w} {h}">\n'
        '<style>.cut{fill:none;stroke:#d00;stroke-width:.25}'
        '.dim{fill:none;stroke:#06c;stroke-width:.22}'
        '.ctr{fill:none;stroke:#777;stroke-width:.15;stroke-dasharray:2,1}'
        '.txt{font-family:Arial,sans-serif;font-size:4px;fill:#111}'
        '.note{font-family:Arial,sans-serif;font-size:3.5px;fill:#111}</style>\n'
    )


def svg_line(x1, y1, x2, y2, cls="cut"):
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>\n'


def svg_rect(x, y, w, h, r=0, cls="cut"):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>\n'


def svg_circle(x, y, r, cls="cut"):
    return f'<circle class="{cls}" cx="{x}" cy="{y}" r="{r}"/>\n'


def svg_text(x, y, text, cls="txt"):
    return f'<text class="{cls}" x="{x}" y="{y}">{text}</text>\n'


def dim_h(x1, x2, y, ref_y, label):
    s = svg_line(x1, ref_y, x1, y, "dim") + svg_line(x2, ref_y, x2, y, "dim")
    s += svg_line(x1, y, x2, y, "dim")
    s += svg_line(x1, y - 1.5, x1 + 3, y, "dim") + svg_line(x1, y + 1.5, x1 + 3, y, "dim")
    s += svg_line(x2, y - 1.5, x2 - 3, y, "dim") + svg_line(x2, y + 1.5, x2 - 3, y, "dim")
    return s + svg_text((x1 + x2) / 2 - len(label), y - 2, label)


def base_svg():
    w, h = 230, 220
    ox, oy = 30, 30
    s = svg_header(w, h)
    s += svg_text(8, 10, "ROV UNIVERSAL PARALLEL GRIPPER - BASE PLATE - SCALE 1:1")
    s += svg_text(8, 17, "Material: 8 mm 6061-T6 aluminum or 10 mm HDPE; deburr all edges")
    s += svg_rect(ox, oy, P["base_w"], P["base_h"], 3)
    for x in (P["mount_x"], P["base_w"] - P["mount_x"]):
        for y in (P["mount_y"], P["base_h"] - P["mount_y"]):
            s += svg_circle(ox + x, oy + y, P["mount_hole_d"] / 2)
    # Rail support holes: four 8.2 mm reamed/clearance holes.
    for x in (P["rail_x0"], P["rail_x1"]):
        for y in (P["base_h"] / 2 - P["rail_spacing"] / 2,
                  P["base_h"] / 2 + P["rail_spacing"] / 2):
            s += svg_circle(ox + x, oy + y, 4.1)
    # Wrist bolt circle.
    cx, cy = ox + P["wrist_center_x"], oy + P["wrist_center_y"]
    s += svg_circle(cx, cy, 30, "ctr")
    s += svg_circle(cx, cy, 8)
    for a in (0, 90, 180, 270):
        rr = P["wrist_bcd"] / 2
        s += svg_circle(cx + rr * cos(a*pi/180), cy + rr * sin(a*pi/180),
                        P["wrist_hole_d"] / 2)
    s += dim_h(ox, ox + P["base_w"], 205, oy + P["base_h"], "170")
    s += svg_text(8, 214, "4x Ø6.5 mounting; 4x Ø8.2 rail; wrist: 4x Ø5.5 on Ø50 PCD + Ø16 center")
    s += "</svg>"
    (OUT / "GRP-001_base_plate_1to1.svg").write_text(s, encoding="utf-8")


def jaw_svg():
    w, h = 340, 220
    s = svg_header(w, h)
    s += svg_text(8, 10, "GRP-002 JAW BLANKS AND REPLACEABLE INSERTS - SCALE 1:1")
    s += svg_text(8, 17, "Jaw: 8 mm 6061-T6; inserts: TPU 95A / NBR rubber, mirror parts as noted")
    # Two jaw blanks; 145 long, 24 wide, three M4 attachment holes.
    for row, y in enumerate((35, 80)):
        s += svg_rect(30, y, P["jaw_len"], P["jaw_w"], 3)
        for x in (42, 92, 142):
            s += svg_circle(x, y + P["jaw_w"]/2, 2.1)
        s += svg_text(185, y + 10, "GRP-002-%s  QTY 1" % ("L" if row == 0 else "R"))
    s += dim_h(30, 175, 115, 104, "145")
    # Bottle V insert: 120 x 18 with 120-degree central shallow V representation.
    y = 135
    s += svg_line(30, y, 150, y)
    s += svg_line(30, y, 30, y + 18)
    s += svg_line(150, y, 150, y + 18)
    s += svg_line(30, y + 18, 75, y + 18)
    s += svg_line(75, y + 18, 90, y + 9)
    s += svg_line(90, y + 9, 105, y + 18)
    s += svg_line(105, y + 18, 150, y + 18)
    s += svg_text(160, y + 10, "GRP-003 bottle/T-plug V insert, QTY 2")
    # Ball cup insert: 70x24, R23 cup notch.
    y2 = 175
    s += svg_rect(30, y2, 70, 24, 2)
    s += f'<path class="cut" d="M 42 {y2} A 23 23 0 0 0 88 {y2}"/>\n'
    s += svg_text(110, y2 + 12, "GRP-004 Ø42 ball cup insert, R23, QTY 2")
    s += svg_text(8, 212, "Insert mounting: 3x M4 through per jaw, 50 mm pitch. Add 2 mm drain grooves.")
    s += "</svg>"
    (OUT / "GRP-002_jaws_and_inserts_1to1.svg").write_text(s, encoding="utf-8")


def dxf_polyline(lines, points, closed=True, layer="CUT"):
    lines.extend(["0", "LWPOLYLINE", "8", layer, "90", str(len(points)), "70", "1" if closed else "0"])
    for x, y in points:
        lines.extend(["10", f"{x:.3f}", "20", f"{y:.3f}"])


def dxf_circle(lines, x, y, r, layer="CUT"):
    lines.extend(["0", "CIRCLE", "8", layer, "10", f"{x:.3f}", "20", f"{y:.3f}", "40", f"{r:.3f}"])


def write_dxf():
    lines = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
             "0", "SECTION", "2", "ENTITIES"]
    dxf_polyline(lines, [(0, 0), (170, 0), (170, 160), (0, 160)])
    for x in (15, 155):
        for y in (15, 145):
            dxf_circle(lines, x, y, 3.25)
    for x in (36, 144):
        for y in (48, 112):
            dxf_circle(lines, x, y, 4.1)
    dxf_circle(lines, 85, 80, 8)
    for a in (0, 90, 180, 270):
        dxf_circle(lines, 85 + 25*cos(a*pi/180), 80 + 25*sin(a*pi/180), 2.75)
    lines += ["0", "ENDSEC", "0", "EOF"]
    (OUT / "GRP-001_base_plate.dxf").write_text("\n".join(lines), encoding="ascii")

    lines = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
             "0", "SECTION", "2", "ENTITIES"]
    for y in (0, 40):
        dxf_polyline(lines, [(0, y), (145, y), (145, y+24), (0, y+24)])
        for x in (12, 62, 112):
            dxf_circle(lines, x, y+12, 2.1)
    lines += ["0", "ENDSEC", "0", "EOF"]
    (OUT / "GRP-002_jaw_blanks.dxf").write_text("\n".join(lines), encoding="ascii")


def pdf_drawing():
    page = landscape(A3)
    c = canvas.Canvas(str(OUT / "ROV_gripper_engineering_drawing_A3.pdf"), pagesize=page)
    font = "Helvetica"
    for candidate in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if Path(candidate).exists():
            try:
                pdfmetrics.registerFont(TTFont("CN", candidate))
                font = "CN"
                break
            except Exception:
                pass
    W, H = page
    mm = 72/25.4

    def tx(x, y, text, size=8):
        c.setFont(font, size)
        c.drawString(x*mm, y*mm, text)

    # Page 1: assembly.
    c.setLineWidth(0.5)
    tx(12, 287, "ROV比赛通用平行夹爪 - 总装图", 16)
    tx(12, 278, "图号 GRP-000 | 单位 mm | 未注公差 ±0.2 | 推荐水下有效载荷 2 kg", 9)
    ox, oy = 45, 55
    c.rect(ox*mm, oy*mm, P["base_w"]*mm, P["base_h"]*mm)
    # Fixed and moving jaws in plan view.
    c.setFillColorRGB(.75, .85, 1)
    c.rect((ox+15)*mm, (oy+60)*mm, 145*mm, 24*mm, fill=1)
    c.rect((ox+15)*mm, (oy+60+120)*mm, 145*mm, 24*mm, fill=1)
    c.setFillColorRGB(0, 0, 0)
    tx(50, 43, "前端/目标物方向 →", 10)
    tx(235, 220, "关键尺寸", 12)
    notes = [
        "最大净开口：120",
        "夹指有效长度：145",
        "基板：170 × 160 × 8",
        "导轨：2 × Ø8，中心距64",
        "丝杆：T8×2，建议POM螺母",
        "目标夹持力：60-100 N",
        "推荐闭合速度：8-15 mm/s",
        "旋腕接口：Ø50 PCD，4×M5",
    ]
    for i, n in enumerate(notes):
        tx(235, 210-i*9, n, 9)
    tx(235, 125, "动作适配", 12)
    for i, n in enumerate([
        "42球：换装R23杯形软指",
        "500ml瓶：120° V形软指",
        "T插头/插销：V形软指前端夹持",
        "旋盖：夹紧T柄后由独立旋腕驱动",
    ]):
        tx(235, 115-i*9, n, 9)
    tx(235, 72, "注意：驱动电机应置于干舱；穿舱轴采用双唇封+油脂腔，或磁耦合。", 8)
    tx(235, 62, "所有外露紧固件使用316不锈钢，并与铝件做电偶腐蚀隔离。", 8)
    c.showPage()

    # Page 2: fabrication and BOM.
    tx(12, 287, "GRP-000 加工与装配要求", 16)
    tx(12, 276, "物料表", 11)
    rows = [
        ("GRP-001", "基板", "6061-T6/HDPE", "170×160×8", "1"),
        ("GRP-002-L/R", "左右夹指骨架", "6061-T6", "145×24×8", "各1"),
        ("GRP-003", "瓶/插头V形软指", "TPU95A/NBR", "120×18×8", "2"),
        ("GRP-004", "球形杯软指", "TPU95A/NBR", "70×24×10, R23", "2"),
        ("STD-01", "直线导轨", "316不锈钢", "Ø8×145", "2"),
        ("STD-02", "梯形丝杆", "316不锈钢", "T8×2×150", "1"),
        ("STD-03", "丝杆螺母", "POM", "T8×2", "1"),
        ("STD-04", "轴承", "陶瓷/316", "608或同等", "2"),
        ("STD-05", "紧固件", "316不锈钢", "M4/M5/M6", "1套"),
    ]
    x = [12, 48, 102, 170, 245, 275]
    headers = ["图号", "名称", "材料", "规格", "数量"]
    y = 265
    for i, h in enumerate(headers):
        tx(x[i]+2, y+2, h, 8)
    c.rect(x[0]*mm, y*mm, (x[-1]-x[0])*mm, 9*mm)
    for xx in x[1:-1]:
        c.line(xx*mm, y*mm, xx*mm, (y+9)*mm)
    for row in rows:
        y -= 9
        c.rect(x[0]*mm, y*mm, (x[-1]-x[0])*mm, 9*mm)
        for xx in x[1:-1]:
            c.line(xx*mm, y*mm, xx*mm, (y+9)*mm)
        for i, value in enumerate(row):
            tx(x[i]+2, y+2, value, 7)
    tx(12, 166, "装配基准与公差", 11)
    requirements = [
        "1. 两根Ø8导轨平行度≤0.10/145；丝杆轴线与导轨平行度≤0.15/145。",
        "2. 夹指安装后两夹持面平行度≤0.5；完全闭合时不得刚性顶死，保留1 mm软垫压缩量。",
        "3. 导轨孔按实际轴径配作；若用直线轴承，按轴承座厂家尺寸修改GRP-001。",
        "4. 夹指边缘倒角C1；所有可能接触赛场器材的边缘覆盖软质材料。",
        "5. 丝杆限位采用两只磁簧/霍尔开关；电流限幅作为第二级夹紧保护。",
        "6. 下水前做30 min空载运行和30 min水下浸泡；不得出现卡滞、松动或进水。",
    ]
    for i, n in enumerate(requirements):
        tx(15, 154-i*11, n, 8)
    tx(12, 80, "建议控制参数", 11)
    for i, n in enumerate([
        "普通抓取：夹紧电流阈值按60 N标定；球形零件使用30-40 N。",
        "插头任务：闭合后保持低电流，不允许继续堵转。",
        "旋盖任务：旋腕限矩建议2-4 N·m，首次以1 N·m开始实物标定。",
        "任何通讯中断：停止丝杆和旋腕，保持机械自锁，不突然张开。",
    ]):
        tx(15, 68-i*10, n, 8)
    c.save()


if __name__ == "__main__":
    base_svg()
    jaw_svg()
    write_dxf()
    pdf_drawing()
    print(f"Generated drawings in: {OUT}")
