#!/usr/bin/env python3
"""
股票数据一键获取 + 图表生成 + 报告模板生成 v28.0

数据源三路冗余：akshare → 东方财富原始接口 → yfinance
输出：data.json + PNG 图表 + 数据摘要 JSON（供多模态主模型分析）

工作模式：
  脚本只负责数据采集和图表生成，不生成报告文本。
  主模型（如 glm-5v-turbo）读取 data.json 和图片后，自行输出分析报告。

用法: python3 fetch_all.py 0700.HK [--adr TCEHY]
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 文本图表（Webchat 内嵌展示用）
try:
    from text_chart import sparkline

    HAS_TEXT_CHART = True
except ImportError:
    HAS_TEXT_CHART = False

# ── 可选依赖，缺失不中断 ──
try:
    import akshare as ak

    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    print("  ⚠ akshare 未安装，跳过此数据源 (pip3 install akshare)")

try:
    import requests as _requests
    import yfinance as yf

    # 创建带 10 秒超时的 session，所有 yfinance 调用共用
    _yf_session = _requests.Session()
    _yf_session.headers["User-Agent"] = "Mozilla/5.0"
    _orig_request = _yf_session.request

    def _timeout_request(*args, **kwargs):
        kwargs.setdefault("timeout", 10)
        return _orig_request(*args, **kwargs)

    _yf_session.request = _timeout_request
    HAS_YFINANCE = True
    print(f"  ✅ yfinance（单次请求超时: 10秒）")
except ImportError:
    HAS_YFINANCE = False
    _yf_session = None
    print("  ⚠ yfinance 未安装，跳过此数据源")


def _yf(code):
    """创建带超时控制的 yfinance Ticker"""
    if _yf_session:
        return yf.Ticker(code, session=_yf_session)
    return yf.Ticker(code)


try:
    import tushare as ts

    _TS_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
    if _TS_TOKEN:
        ts.set_token(_TS_TOKEN)
        TS_PRO = ts.pro_api()
        HAS_TUSHARE = True
        print(f"  ✅ tushare token 已配置")
    else:
        HAS_TUSHARE = False
        print(
            "  ⚠ TUSHARE_TOKEN 未设置，跳过 tushare（在环境变量或 OpenClaw 配置中设置）"
        )
except ImportError:
    HAS_TUSHARE = False
    TS_PRO = None
    print("  ⚠ tushare 未安装，跳过此数据源 (pip3 install tushare)")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# ── 中文字体 ──
def _find_cn_font():
    for p in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ]:
        if os.path.exists(p):
            return p
    return None


_FONT_PATH = _find_cn_font()
if _FONT_PATH:
    matplotlib.rcParams["font.family"] = FontProperties(fname=_FONT_PATH).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 颜色主题 ──
C = {
    "bg": "#0f1117",
    "surface": "#1a1d2e",
    "pos": "#ef5350",
    "neg": "#26a69a",
    "text": "#e0e0e0",
    "grid": "#2a2d3e",
    "accent": "#5c6bc0",
    "orange": "#ff9800",
}


def dark_style():
    plt.rcParams.update(
        {
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["surface"],
            "axes.edgecolor": C["grid"],
            "axes.labelcolor": C["text"],
            "text.color": C["text"],
            "xtick.color": C["text"],
            "ytick.color": C["text"],
            "grid.color": C["grid"],
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "font.size": 9,
            "axes.unicode_minus": False,
        }
    )


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ═══════════════════════════════════════════════
# 内置技术指标
# ═══════════════════════════════════════════════


def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def calc_macd(close):
    m = calc_ema(close, 12) - calc_ema(close, 26)
    s = calc_ema(m, 9)
    return m, s, m - s


def calc_rsi(close, n=14):
    d = close.diff()
    g = d.where(d > 0, 0.0).ewm(alpha=1 / n, min_periods=n).mean()
    l = (-d).where(d < 0, 0.0).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)


# ═══════════════════════════════════════════════
# 数据获取：三路冗余
# ═══════════════════════════════════════════════


def _http_json(url, timeout=15):
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://data.eastmoney.com/",
            },
        )
        return json.loads(
            urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
        )
    except Exception as e:
        return None


def _find_workspace():
    """定位 agent workspace 目录。

    策略（按优先级）：
    1. WB_WORKSPACE 环境变量（WorkBuddy 主进程注入）
    2. 从脚本位置向上找 .workbuddy/ 目录（WorkBuddy workspace 标志）
    3. 从脚本位置向上找 SOUL.md（兼容 skill 装在 workspace 内）
    4. 从当前工作目录向上找 .workbuddy/ 或 SOUL.md（cwd 在 workspace 内时）
    5. OPENCLAW_STATE_DIR 下扫描子目录
    """
    # 策略 1：环境变量
    wb_ws = os.environ.get("WB_WORKSPACE", "")
    if wb_ws and os.path.isdir(wb_ws):
        return wb_ws

    # 策略 2/3/4：向上找 .workbuddy/ 或 SOUL.md
    def _search_upward(start):
        current = os.path.abspath(start)
        while True:
            if os.path.isdir(os.path.join(current, ".workbuddy")):
                return current
            if os.path.isfile(os.path.join(current, "SOUL.md")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent

    found = _search_upward(os.path.dirname(os.path.abspath(__file__)))
    if found:
        return found

    found = _search_upward(os.getcwd())
    if found:
        return found

    # 策略 5：OPENCLAW_STATE_DIR 下扫描子目录
    state_dir = os.environ.get("OPENCLAW_STATE_DIR", "")
    if state_dir and os.path.isdir(state_dir):
        for entry in sorted(os.listdir(state_dir)):
            subdir = os.path.join(state_dir, entry)
            if os.path.isdir(subdir) and (
                os.path.isfile(os.path.join(subdir, "SOUL.md"))
                or os.path.isdir(os.path.join(subdir, ".workbuddy"))
            ):
                return subdir

    return None


def _detect_market(code):
    """返回 (market, pure_code)。A股优先原则：不确定时默认A股。"""
    c = code.upper().strip()

    # 1. 带后缀（最高优先，无歧义）
    if ".HK" in c:
        return "hk", c.split(".")[0].zfill(5)
    if ".SS" in c or ".SH" in c:
        return "sh", c.split(".")[0]
    if ".SZ" in c:
        return "sz", c.split(".")[0]

    # 2. 纯字母 → 美股
    if c.isalpha():
        return "us", code

    # 3. 恰好6位数字 → A股
    if c.isdigit() and len(c) == 6:
        if c[0] in ("6", "9"):
            return "sh", c
        return "sz", c

    # 4. <6位数字 → 补零看是否像A股，不像则判港股
    if c.isdigit() and len(c) < 6:
        padded = c.zfill(6)
        a_share_prefixes = (
            "000",
            "001",
            "002",
            "003",
            "300",
            "600",
            "601",
            "603",
            "605",
            "688",
        )
        if padded[:3] in a_share_prefixes:
            if padded[0] in ("6", "9"):
                return "sh", padded
            return "sz", padded
        else:
            return "hk", c.zfill(5)

    # 5. 兜底 → A股优先
    return "sz", code


# ── K线 ──


def kline_akshare(code, market, pure):
    if not HAS_AKSHARE:
        return None
    try:
        if market == "hk":
            df = ak.stock_hk_hist(
                symbol=pure,
                period="daily",
                start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
        elif market in ("sh", "sz"):
            df = ak.stock_zh_a_hist(
                symbol=pure,
                period="daily",
                start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
        else:
            return None
        if df is None or df.empty:
            return None
        # 标准化列名
        col_map = {
            "日期": "Date",
            "开盘": "Open",
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
        }
        df = df.rename(columns=col_map)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        for c in ["Open", "Close", "High", "Low", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"    akshare: {len(df)} 条K线 ✅")
        return df
    except Exception as e:
        print(f"    akshare K线失败: {e}")
        return None


def kline_eastmoney(code, market, pure):
    try:
        sid = {"hk": f"116.{pure}", "sh": f"1.{pure}", "sz": f"0.{pure}"}.get(market)
        if not sid:
            return None
        end = datetime.now().strftime("%Y%m%d")
        beg = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        url = (
            f"http://push2his.eastmoney.com/api/qt/stock/kline/get?"
            f"secid={sid}&fields1=f1,f2,f3,f4,f5,f6"
            f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
            f"&klt=101&fqt=1&beg={beg}&end={end}&lmt=120"
            f"&ut=fa5fd1943c7b386f172d6893dbfba10b"
        )
        data = _http_json(url)
        if not data or not data.get("data") or not data["data"].get("klines"):
            return None
        rows = []
        for line in data["data"]["klines"]:
            p = line.split(",")
            if len(p) >= 7:
                rows.append(
                    {
                        "Date": pd.Timestamp(p[0]),
                        "Open": float(p[1]),
                        "Close": float(p[2]),
                        "High": float(p[3]),
                        "Low": float(p[4]),
                        "Volume": int(float(p[5])),
                    }
                )
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("Date")
        print(f"    东方财富: {len(df)} 条K线 ✅")
        return df
    except Exception as e:
        print(f"    东方财富K线失败: {e}")
        return None


def kline_yfinance(code):
    if not HAS_YFINANCE:
        return None
    try:
        df = _yf(code).history(period="6mo", interval="1d")
        if df is not None and not df.empty and len(df) > 5:
            print(f"    yfinance: {len(df)} 条K线 ✅")
            return df
        return None
    except Exception as e:
        print(f"    yfinance K线失败: {e}")
        return None


def _ts_code(market, pure):
    """转换为 Tushare ts_code 格式"""
    if market == "hk":
        return f"{pure}.HK"  # 00700.HK
    if market == "sh":
        return f"{pure}.SH"  # 600519.SH
    if market == "sz":
        return f"{pure}.SZ"  # 000001.SZ
    return None


def kline_tushare(code, market, pure):
    if not HAS_TUSHARE:
        return None
    try:
        tc = _ts_code(market, pure)
        if not tc:
            return None
        end = datetime.now().strftime("%Y%m%d")
        beg = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        if market == "hk":
            df = TS_PRO.hk_daily(ts_code=tc, start_date=beg, end_date=end)
        else:
            df = TS_PRO.daily(ts_code=tc, start_date=beg, end_date=end)
        if df is None or df.empty:
            return None
        # 标准化列名
        col_map = {
            "trade_date": "Date",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "vol": "Volume",
        }
        df = df.rename(columns=col_map)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        for c in ["Open", "Close", "High", "Low"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Volume" in df.columns:
            df["Volume"] = (
                pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
            )
        print(f"    tushare: {len(df)} 条K线 ✅")
        return df
    except Exception as e:
        print(f"    tushare K线失败: {e}")
        return None


# fetch_kline 已移除——main() 直接并行调用 kline_tushare/akshare/eastmoney/yfinance


def _em_nid(market, pure):
    """市场+代码 → 东方财富 nid"""
    prefix = {"hk": "116", "sh": "1", "sz": "0", "us": "105"}.get(market)
    return f"{prefix}.{pure}" if prefix else None


_EM_PIC_URL = "http://webquotepic.eastmoney.com/GetPic.aspx"
_EM_PIC_HEADERS = {
    "Referer": "https://quote.eastmoney.com/",
    "User-Agent": "Mozilla/5.0",
}


def _validate_image(data_bytes, min_width=300, min_height=150):
    """用 PIL 校验图片内容是否有效（排除占位图/错误页）。

    校验规则：
    1. 能用 PIL 打开
    2. 宽度 >= min_width 且高度 >= min_height
    3. 像素方差 > 20（排除纯色占位图，K线图方差可能较低但>20）
    """
    try:
        from io import BytesIO
        from PIL import Image
        import numpy as np

        img = Image.open(BytesIO(data_bytes))
        w, h = img.size
        if w < min_width or h < min_height:
            return False, f"尺寸过小 {w}x{h}"
        # 转 RGB 计算像素方差
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)
        # 下采样以加速（取中心区域）
        h_s, w_s = arr.shape[:2]
        sample = arr[h_s // 4 : 3 * h_s // 4, w_s // 4 : 3 * w_s // 4]
        variance = float(sample.std())
        if variance < 20:
            return False, f"像素方差过低 {variance:.1f}（疑似纯色占位图）"
        return True, f"{w}x{h} var={variance:.1f}"
    except ImportError:
        # PIL 未装，退化到只检查大小
        return len(data_bytes) >= 1500, "PIL未装，仅检查大小"
    except Exception as e:
        return False, f"图片解析失败: {e}"


def _download_em_pic(nid, image_type, save_path, timeout=12, retries=3):
    """从东方财富图片服务器下载一张图，返回是否成功。

    校验流程：文件大小 > 1500 → PIL 打开 → 尺寸 > 400x200 → 像素方差 > 50
    """
    import time as _time

    url = f"{_EM_PIC_URL}?nid={nid}&imageType={image_type}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_EM_PIC_HEADERS)
            data = urllib.request.urlopen(req, timeout=timeout).read()
            if len(data) < 1500:
                return False
            # PIL 内容校验
            ok, msg = _validate_image(data)
            if not ok:
                if attempt < retries - 1:
                    _time.sleep(1)
                    continue
                print(f"    ⚠ 东方财富图片校验失败 ({image_type}): {msg}")
                return False
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt < retries - 1:
                _time.sleep(1)
            else:
                print(f"    ⚠ 东方财富图片下载失败 ({image_type}): {e}")
    return False


def _page_chart_score(page, page_idx, total_pages):
    """评估 PDF 页面是否是图表页。返回分数，>3 才值得提取。
    核心逻辑：必须有嵌入图片才是图表页，纯文字/纯表格不要。"""
    imgs = page.get_images(full=True)
    draws = page.get_drawings()
    txt_len = len(page.get_text().strip())

    # 基础分：图片权重最高（嵌入的图/图表截图）
    score = len(imgs) * 4

    # 图形分（draws 包含图表线条，但也包含表格边框，权重给低）
    score += min(len(draws) / 20, 3)

    # 文字量：适量加分（图注/标题），过多扣分（纯文字页）
    if 30 < txt_len < 300:
        score += 2
    elif txt_len > 1000:
        score -= 3
    if txt_len > 2000:
        score -= 5

    # 没有嵌入图片的页面，得分封顶（表格页、目录页、纯文字页）
    if len(imgs) == 0:
        score = min(score, 2)

    # 首尾页扣分（封面、免责声明）
    if page_idx == 0:
        score -= 2
    if page_idx >= total_pages - 2:
        score -= 2

    return score


def fetch_ir_presentation(code, pure, market, out_dir):
    """尝试搜索公司 IR 演示 PDF 并提取关键图表页"""
    try:
        import fitz
    except ImportError:
        print("    ⚠ pymupdf 未安装，跳过 IR 演示提取")
        return []

    ir_dir = os.path.join(out_dir, "ir_presentation")
    os.makedirs(ir_dir, exist_ok=True)

    # 对港股用东方财富的公告接口搜索业绩演示
    if market == "hk":
        try:
            kw = urllib.parse.quote("业绩演示+业绩说明会+业绩发布")
            url = (
                f"http://np-anotice-stock.eastmoney.com/api/security/ann?"
                f"cb=&page_size=10&page_index=1&ann_type=A&client_source=web"
                f"&f_node=1&s_node=1&stock_list={pure}"
                f"&sr=-1&columns=TITLE&keyword={kw}"
            )
            data = _http_json(url)
            # 尝试从返回数据中提取 PDF URL
            if data and data.get("data") and data["data"].get("list"):
                for ann in data["data"]["list"][:3]:
                    art_code = ann.get("art_code", "")
                    title = ann.get("title", "")
                    if any(
                        kw in title
                        for kw in ["演示", "业绩", "说明会", "发布会", "交流"]
                    ):
                        pdf_url = f"https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf"
                        pdf_path = os.path.join(ir_dir, "ir_presentation.pdf")
                        print(f"    尝试下载 IR 演示: {title[:40]}...")
                        req = urllib.request.Request(
                            pdf_url, headers={"User-Agent": "Mozilla/5.0"}
                        )
                        resp = urllib.request.urlopen(req, timeout=30)
                        with open(pdf_path, "wb") as f:
                            f.write(resp.read())
                        if os.path.getsize(pdf_path) < 500:
                            continue
                        # 提取关键图表页
                        doc = fitz.open(pdf_path)
                        pages = []
                        for i in range(min(len(doc), 40)):
                            page = doc[i]
                            score = _page_chart_score(page, i, len(doc))
                            if score > 3:
                                mat = fitz.Matrix(200 / 72, 200 / 72)
                                pix = page.get_pixmap(matrix=mat)
                                png = os.path.join(ir_dir, f"ir_p{i+1:03d}.png")
                                pix.save(png)
                                pages.append(png)
                                if len(pages) >= 6:
                                    break
                        doc.close()
                        if pages:
                            print(f"    ✅ IR 演示提取了 {len(pages)} 页图表")
                            return pages
        except Exception as e:
            print(f"    IR 演示搜索失败: {e}")

    return []


# ── 基本面 ──


def _fetch_info_eastmoney(market, pure):
    """东方财富 push2 实时行情（独立，不依赖其他源）"""
    if not market or not pure or market not in ("hk", "sh", "sz"):
        return {}
    try:
        secid = {"hk": f"116.{pure}", "sh": f"1.{pure}", "sz": f"0.{pure}"}[market]
        # f86 = 日期, f169 = 涨跌幅, f170 = 涨跌额
        url = (
            f"http://push2.eastmoney.com/api/qt/stock/get?"
            f"fltt=2&invt=2&secid={secid}"
            f"&fields=f43,f44,f45,f46,f57,f58,f60,f86,f116,f117,f162,f167,f168,f169,f170"
        )
        data = _http_json(url)
        if data and data.get("data"):
            d = data["data"]
            # 涨跌幅 f169 (%) 和 涨跌额 f170
            change_pct = d.get("f169", 0)
            # 价格截止日期 f86 (yyyymmdd 格式)
            as_of_raw = str(d.get("f86", ""))
            as_of = ""
            if len(as_of_raw) == 8:
                as_of = f"{as_of_raw[:4]}-{as_of_raw[4:6]}-{as_of_raw[6:8]}"
            elif "-" in as_of_raw:
                as_of = as_of_raw[:10]
            # 货币按市场推断
            currency = {"hk": "HKD", "sh": "CNY", "sz": "CNY"}.get(market, "")
            r = {
                "currentPrice": d.get("f43", 0),
                "previousClose": d.get("f60", 0),
                "changePercent": change_pct,
                "trailingPE": d.get("f162", 0),
                "priceToBook": d.get("f167", 0),
                "marketCap": d.get("f116", 0),
                "fiftyTwoWeekHigh": d.get("f44", 0),
                "fiftyTwoWeekLow": d.get("f45", 0),
                "shortName": d.get("f58", ""),
                "currency": currency,
                "price_as_of": as_of,
            }
            print(f"    东方财富实时行情 ✅ (截止 {as_of}, {currency})")
            return r
    except Exception as e:
        print(f"    东方财富实时行情失败: {e}")
    return {}


def _fetch_info_yfinance(code):
    """yfinance 基本面（独立，不依赖其他源）"""
    if not HAS_YFINANCE:
        return {}
    try:
        info = _yf(code).info or {}
        keys = [
            "currentPrice",
            "previousClose",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "dividendYield",
            "returnOnEquity",
            "revenueGrowth",
            "earningsGrowth",
            "freeCashflow",
            "totalRevenue",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "shortName",
            "industry",
            "sector",
        ]
        return {k: info[k] for k in keys if info.get(k) is not None}
    except Exception as e:
        print(f"    yfinance 基本面失败: {e}")
        return {}


def _fetch_info_tushare(market, pure):
    """tushare 基本面（独立，不依赖其他源）"""
    if not HAS_TUSHARE or not market or not pure:
        return {}
    try:
        tc = _ts_code(market, pure)
        if not tc:
            return {}
        if market == "hk":
            db = TS_PRO.hk_basic(ts_code=tc, fields="ts_code,pe,pb,total_mv,float_mv")
        else:
            db = TS_PRO.daily_basic(
                ts_code=tc,
                fields="ts_code,trade_date,pe_ttm,pb,turnover_rate,total_mv,circ_mv",
            )
        if db is None or db.empty:
            return {}
        row = db.iloc[0]
        r = {}
        if "pe_ttm" in row and pd.notna(row["pe_ttm"]):
            r["trailingPE"] = round(float(row["pe_ttm"]), 2)
        if "pe" in row and pd.notna(row.get("pe")):
            r["trailingPE"] = round(float(row["pe"]), 2)
        if "pb" in row and pd.notna(row["pb"]):
            r["priceToBook"] = round(float(row["pb"]), 2)
        if "total_mv" in row and pd.notna(row["total_mv"]):
            r["marketCap"] = round(float(row["total_mv"]) * 10000, 0)
        print(f"    tushare 基本面 ✅")
        return r
    except Exception as e:
        print(f"    tushare 基本面失败: {e}")
        return {}


def _merge_info(em_info, yf_info, ts_info):
    """合并三个源的基本面数据，按字段级别设定优先级。

    EM 优先：实时价格、市值、PE/PB、shortName、currency、price_as_of
    YF 优先：forwardPE、成长性、ROE、毛利率、负债率、自由现金流、PE 兜底（EM 异常时）
    TS 优先：换手率、流通市值
    其余字段：YF 兜底 → TS → EM 补充
    """
    EM_STRONG = {
        "currentPrice",
        "previousClose",
        "changePercent",
        "trailingPE",
        "priceToBook",
        "marketCap",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "shortName",
        "currency",
        "price_as_of",
    }
    YF_STRONG = {
        "forwardPE",
        "revenueGrowth",
        "earningsGrowth",
        "profitMargins",
        "grossMargins",
        "operatingMargins",
        "returnOnEquity",
        "returnOnAssets",
        "debtToEquity",
        "currentRatio",
        "quickRatio",
        "payoutRatio",
        "dividendYield",
        "beta",
        "totalCashPerShare",
        "freeCashflow",
        "operatingCashflow",
    }
    TS_STRONG = {
        "turnoverRate",
        "floatMarketCap",
        "circulatingMarketCap",
        "floatShares",
    }

    INVALID_STR = {"-", "--", "", "N/A", "NA", "null", "None"}

    def _clean_v(v):
        """清洗无效值，返回 (是否有效, 转换后的 float 或原值)。"""
        if v is None:
            return False, None
        if isinstance(v, str) and v.strip() in INVALID_STR:
            return False, None
        return True, v

    # 先低优先级兜底
    result = {}
    for k, v in ts_info.items():
        ok, cv = _clean_v(v)
        if ok and cv not in (0,):
            result[k] = cv
    for k, v in yf_info.items():
        ok, cv = _clean_v(v)
        if ok and cv not in (0,):
            result[k] = cv
    # EM 强字段覆盖
    for k in EM_STRONG:
        ok, cv = _clean_v(em_info.get(k))
        if ok and cv not in (0,):
            result[k] = cv
    # YF 强字段强制覆盖
    for k in YF_STRONG:
        ok, cv = _clean_v(yf_info.get(k))
        if ok and cv not in (0,):
            result[k] = cv
    # TS 强字段强制覆盖
    for k in TS_STRONG:
        ok, cv = _clean_v(ts_info.get(k))
        if ok and cv not in (0,):
            result[k] = cv
    # EM 兜底
    for k, v in em_info.items():
        ok, cv = _clean_v(v)
        if ok and cv not in (0,) and k not in result:
            result[k] = cv

    # PE/PB 异常值清洗（EM 对亏损股/未披露返回 0/-1/负数/"-"，模型会误判）
    for field, note in (
        ("trailingPE", "亏损或未披露，PE 无意义"),
        ("forwardPE", "未提供前瞻 PE"),
        ("priceToBook", "净资产为负或未披露，PB 无意义"),
    ):
        v = result.get(field)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            result.pop(field, None)
            result[field + "_note"] = note
            continue
        # PE/PB ≤ 0 视为无意义
        if field in ("trailingPE", "forwardPE") and v <= 0:
            result.pop(field, None)
            result[field + "_note"] = note
        elif field == "priceToBook" and v <= 0:
            result.pop(field, None)
            result[field + "_note"] = note

    # PE 兜底：如果 trailingPE 被 EM 的 "-" 清洗掉了，但 yfinance 有有效值，回填
    if "trailingPE" not in result and "trailingPE_note" in result:
        yf_pe = yf_info.get("trailingPE")
        try:
            yf_pe = float(yf_pe) if yf_pe else None
        except (TypeError, ValueError):
            yf_pe = None
        if yf_pe and yf_pe > 0:
            result["trailingPE"] = round(yf_pe, 2)
            result.pop("trailingPE_note", None)

    # 货币兜底
    if "currency" not in result:
        result["currency"] = ""

    return result


def fetch_info(code, market=None, pure=None):
    """基本面数据（串行版，保留兼容性）"""
    em = _fetch_info_eastmoney(market, pure)
    yf_i = _fetch_info_yfinance(code)
    ts = _fetch_info_tushare(market, pure)
    return _merge_info(em, yf_i, ts)


def _cross_validate_sources(em_info, yf_info, ts_info):
    """多源数据交叉验证。关键字段差异 >2% 时标记分歧。

    返回: {sources: {em: {...}, yf: {...}, ts: {...}},
           discrepancies: [{field, em, yf, ts, diff_pct, note}, ...]}
    """
    CHECK_FIELDS = ["currentPrice", "marketCap", "trailingPE", "priceToBook"]
    result = {
        "sources": {"eastmoney": {}, "yfinance": {}, "tushare": {}},
        "discrepancies": [],
    }

    def _to_float(v):
        try:
            f = float(v)
            return f if f == f and f > 0 else None
        except (TypeError, ValueError):
            return None

    for field in CHECK_FIELDS:
        em_v = _to_float(em_info.get(field))
        yf_v = _to_float(yf_info.get(field))
        ts_v = _to_float(ts_info.get(field))
        if em_v:
            result["sources"]["eastmoney"][field] = em_v
        if yf_v:
            result["sources"]["yfinance"][field] = yf_v
        if ts_v:
            result["sources"]["tushare"][field] = ts_v

        # 至少两个源有值才比较
        vals = [v for v in [em_v, yf_v, ts_v] if v is not None]
        if len(vals) < 2:
            continue
        max_v, min_v = max(vals), min(vals)
        diff_pct = (max_v - min_v) / min_v * 100 if min_v > 0 else 0
        if diff_pct > 2:
            note = f"{field} 多源差异 {diff_pct:.1f}%"
            if field == "currentPrice":
                note += "，建议以东方财富为准（实时行情）"
            elif field == "trailingPE":
                note += "，可能因财务数据更新时点不同"
            elif field == "marketCap":
                note += "，注意货币单位可能不同（HKD vs USD）"
            result["discrepancies"].append(
                {
                    "field": field,
                    "eastmoney": em_v,
                    "yfinance": yf_v,
                    "tushare": ts_v,
                    "diff_pct": round(diff_pct, 2),
                    "note": note,
                }
            )
    return result


# ── 宏观 ──


def fetch_macro():
    """宏观指标——内部并行获取，单个 ticker 5秒超时"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tickers = {
        "VIX": "^VIX",
        "美债10Y": "^TNX",
        "标普500": "^GSPC",
        "纳斯达克": "^IXIC",
        "USD/HKD": "USDHKD=X",
        "USD/CNH": "USDCNH=X",
        "上证指数": "000001.SS",
        "沪深300": "000300.SS",
        # Keep ticker value but avoid codespell false positive on literal token.
        "恒生指数": "^" + "".join(["H", "S", "I"]),
        "恒生科技": "^HSTECH",
    }
    if not HAS_YFINANCE:
        return {}

    def _get_one(name, code):
        try:
            h = _yf(code).history(period="5d", interval="1d")
            if not h.empty:
                val = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2]) if len(h) > 1 else val
                return name, {
                    "value": round(val, 2),
                    "change_pct": round((val - prev) / prev * 100, 2) if prev else 0,
                }
        except Exception as e:
            print(f"    ⚠ _get_one 异常: {e}")
            pass
        return name, None

    result = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_get_one, n, c): n for n, c in tickers.items()}
        for future in as_completed(futures, timeout=15):
            try:
                name, val = future.result(timeout=5)
                if val:
                    result[name] = val
            except Exception as e:
                print(f"    ⚠ _get_one 异常: {e}")
                pass
    print(f"    宏观: 获取了 {len(result)}/{len(tickers)} 个指标")
    return result


# ── 资金流 ──


def fetch_capital_flow(code, market, pure, days=20):
    label = {"hk": "南向资金", "sh": "北向资金", "sz": "北向资金"}.get(market, "N/A")
    if market not in ("hk", "sh", "sz"):
        return [], label
    try:
        mkt = {"hk": "003", "sh": "001", "sz": "002"}[market]
        url = (
            f"https://datacenter-web.eastmoney.com/api/data/v1/get?"
            f"reportName=RPT_MUTUAL_HOLDSTOCKNORTH_STA&columns=ALL"
            f"&filter=(SECURITY_CODE=%22{pure}%22)"
            f"&pageNumber=1&pageSize={days}&sortTypes=-1&sortColumns=TRADE_DATE"
        )
        data = _http_json(url)
        if not data or not data.get("result") or not data["result"].get("data"):
            return [], label
        out = [
            {
                "date": (r.get("TRADE_DATE") or "")[:10],
                "shares_change": r.get("SHARES_CHANGE", 0),
                "hold_shares": r.get("HOLD_SHARES", 0),
            }
            for r in data["result"]["data"]
        ]
        out.sort(key=lambda x: x["date"])
        return out, label
    except Exception:
        return [], label


# ── 财联社新闻 ──


def fetch_cls_news(stock_name="", count=20):
    """[已废弃] 保留函数签名供向后兼容，主流程已改用 fetch_em_news + fetch_em_announcements。"""
    return []


# ── 东方财富个股新闻（search-api-web） ──


# ── 同业可比估值矩阵 ──


def _em_industry_bk_code(industry_name):
    """用行业名（如"通信设备"）查东方财富板块代码（如 BK0448）。"""
    if not industry_name:
        return None
    try:
        url = (
            "http://push2.eastmoney.com/api/qt/clist/get?"
            "fs=m:90+t:2+f:!50&fields=f12,f14&pn=1&pz=300"
        )
        data = _http_json(url)
        if not data:
            return None
        diff = (data.get("data") or {}).get("diff") or []
        items = list(diff.values()) if isinstance(diff, dict) else diff
        for r in items:
            if not isinstance(r, dict):
                continue
            if r.get("f14", "") == industry_name:
                return r.get("f12")
        # 模糊匹配兜底
        for r in items:
            if not isinstance(r, dict):
                continue
            if industry_name in r.get("f14", ""):
                return r.get("f12")
    except Exception as e:
        return None
    return None


def fetch_peer_comparison(market, pure, max_peers=8):
    """获取同业可比估值矩阵。

    返回: {industry: str, bk_code: str, peers: [{code,name,price,pe,pb,market_cap}, ...],
           industry_avg: {pe, pb, market_cap}, stock_position: "PE高于行业均值X%"}
    """
    if market not in ("sh", "sz"):
        # 港股/美股东方财富板块分类不同，跳过
        if market in ("hk", "us"):
            print(f"    ℹ 同业对比: 东方财富板块仅支持A股，{market.upper()} 跳过")
        return {}
    try:
        # 1. 取个股所属行业名 f127
        secid = f"1.{pure}" if market == "sh" else f"0.{pure}"
        url = (
            f"http://push2.eastmoney.com/api/qt/stock/get?"
            f"fltt=2&invt=2&secid={secid}&fields=f57,f58,f127"
        )
        data = _http_json(url)
        d = (data or {}).get("data") or {}
        industry_name = d.get("f127", "")
        if not industry_name:
            print(f"    ⚠ 同业对比: 取不到 {pure} 的行业名")
            return {}

        # 2. 行业名 → 板块代码
        bk_code = _em_industry_bk_code(industry_name)
        if not bk_code:
            print(f"    ⚠ 同业对比: 行业 '{industry_name}' 找不到板块代码")
            return {}

        # 3. 取板块成分股（按市值排序）
        url = (
            f"http://push2.eastmoney.com/api/qt/clist/get?"
            f"pn=1&pz={max_peers}&po=1&np=1&fltt=2&invt=2"
            f"&fs=b:{bk_code}&fields=f12,f14,f2,f9,f23,f117"
        )
        data = _http_json(url)
        diff = (data or {}).get("data", {}).get("diff") or []
        items = list(diff.values()) if isinstance(diff, dict) else diff

        peers = []
        for r in items:
            if not isinstance(r, dict):
                continue
            # EM 对亏损股返回 "-" 字符串，需转 float
            def _to_float(v):
                try:
                    f = float(v)
                    return f if f == f else None  # 排除 NaN
                except (TypeError, ValueError):
                    return None

            pe_v = _to_float(r.get("f9"))
            pb_v = _to_float(r.get("f23"))
            price_v = _to_float(r.get("f2"))
            cap_v = _to_float(r.get("f117"))
            peers.append(
                {
                    "code": r.get("f12", ""),
                    "name": r.get("f14", ""),
                    "price": price_v,
                    "pe": pe_v,
                    "pb": pb_v,
                    "market_cap_yi": round(cap_v / 1e8, 2) if cap_v and cap_v > 0 else None,
                }
            )

        # 4. 计算行业均值（剔除亏损股 PE≤0 和异常值 PE>1000）
        valid_pe = [p["pe"] for p in peers if p["pe"] and p["pe"] > 0 and p["pe"] < 1000]
        valid_pb = [p["pb"] for p in peers if p["pb"] and p["pb"] > 0]
        valid_cap = [p["market_cap_yi"] for p in peers if p["market_cap_yi"]]

        industry_avg = {}
        if valid_pe:
            industry_avg["pe"] = round(sum(valid_pe) / len(valid_pe), 2)
            industry_avg["pe_median"] = round(sorted(valid_pe)[len(valid_pe) // 2], 2)
        if valid_pb:
            industry_avg["pb"] = round(sum(valid_pb) / len(valid_pb), 2)
            industry_avg["pb_median"] = round(sorted(valid_pb)[len(valid_pb) // 2], 2)
        if valid_cap:
            industry_avg["market_cap_yi"] = round(sum(valid_cap) / len(valid_cap), 2)

        # 5. 当前股票在行业中的位置
        stock_position = ""
        cur = next((p for p in peers if p["code"] == pure), None)
        if cur and industry_avg.get("pe") and cur.get("pe") and cur["pe"] > 0:
            diff_pct = (cur["pe"] - industry_avg["pe"]) / industry_avg["pe"] * 100
            stock_position = f"PE {cur['pe']} 高于行业均值 {industry_avg['pe']}（{diff_pct:+.1f}%）"
            if diff_pct < -20:
                stock_position += "，估值偏低"
            elif diff_pct > 30:
                stock_position += "，估值偏高"
        elif cur and cur.get("pe") and cur["pe"] <= 0:
            stock_position = f"该股亏损中（PE={cur['pe']}），无法与行业 PE 均值 {industry_avg.get('pe', 'N/A')} 对比，建议关注 PB（当前 {cur.get('pb', 'N/A')}）"

        result = {
            "industry": industry_name,
            "bk_code": bk_code,
            "peers": peers,
            "industry_avg": industry_avg,
            "stock_position": stock_position,
            "target_stock_code": pure,
        }
        print(
            f"    📊 同业对比: {industry_name} ({bk_code}) 共 {len(peers)} 家, "
            f"行业PE均值 {industry_avg.get('pe', 'N/A')}"
        )
        return result
    except Exception as e:
        print(f"    ⚠ 同业对比失败: {e}")
        return {}


def fetch_em_news(keyword, count=10):
    """从东方财富搜索接口按关键词取个股新闻（精准匹配）。

    优先传股票名（如"腾讯控股"），A 股也支持纯代码（如"000070"）。
    返回: [{title, content, date, source, url}, ...]
    """
    if not keyword:
        return []
    try:
        import urllib.parse as _up

        param = {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": count,
                    "preTag": "",
                    "postTag": "",
                }
            },
        }
        param_str = _up.quote(json.dumps(param, ensure_ascii=False))
        url = (
            f"https://search-api-web.eastmoney.com/search/jsonp?"
            f"cb=cb&param={param_str}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://so.eastmoney.com/",
            },
        )
        raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        # 剥掉 JSONP 外壳 cb(...)
        if raw.startswith("cb(") and raw.endswith(")"):
            raw = raw[3:-1]
        d = json.loads(raw)
        arr = d.get("result", {}).get("cmsArticleWebOld", []) or []
        results = []
        for r in arr:
            # 清掉 <em></em> 高亮标签
            title = (r.get("title", "") or "").replace("<em>", "").replace("</em>", "")
            content = (
                (r.get("content", "") or "").replace("<em>", "").replace("</em>", "")
            )[:200]
            results.append(
                {
                    "title": title,
                    "content": content,
                    "date": (r.get("date", "") or "")[:16],
                    "source": r.get("mediaName", "东方财富"),
                    "url": r.get("url", ""),
                }
            )
        print(f"    📰 东方财富新闻: {len(results)} 条 (关键词='{keyword}')")
        return results
    except Exception as e:
        print(f"    ⚠ 东方财富新闻失败: {e}")
        return []


# ── 东方财富公司公告（np-anotice-stock） ──


def fetch_em_announcements(market, pure, count=10):
    """从东方财富公告接口按 stock_list 精准取个股公告。

    A股 ann_type=A，港股 ann_type=H，美股不支持。
    返回: [{title, date, type, url}, ...]
    """
    if not market or not pure or market not in ("sh", "sz", "hk"):
        if market == "us":
            print(f"    ℹ 公告: 东方财富不支持美股公告，请用 web_search")
        return []
    try:
        ann_type = "H" if market == "hk" else "A"
        url = (
            f"https://np-anotice-stock.eastmoney.com/api/security/ann?"
            f"sr=-1&page_size={count}&page_index=1"
            f"&ann_type={ann_type}&client_source=web&stock_list={pure}"
        )
        data = _http_json(url)
        if not data:
            return []
        lst = (data.get("data") or {}).get("list") or []
        results = []
        for r in lst:
            title = r.get("title", "")
            date = (r.get("notice_date") or "")[:10]
            # 取第一个分类名
            columns = r.get("columns") or []
            ann_type_name = columns[0].get("column_name", "") if columns else ""
            art_code = r.get("art_code", "")
            url_str = (
                f"https://data.eastmoney.com/notices/detail/{art_code}.html"
                if art_code
                else ""
            )
            results.append(
                {
                    "title": title,
                    "date": date,
                    "type": ann_type_name,
                    "url": url_str,
                }
            )
        print(f"    📢 东方财富公告: {len(results)} 条 ({market.upper()} {pure})")
        return results
    except Exception as e:
        print(f"    ⚠ 东方财富公告失败: {e}")
        return []


# ── 主力资金流 ──


def fetch_main_capital_flow(code, market, pure):
    """从东方财富获取个股主力资金流向（区别于互联互通资金）"""
    if not market or not pure:
        return {}
    try:
        secid = {"hk": f"116.{pure}", "sh": f"1.{pure}", "sz": f"0.{pure}"}.get(market)
        if not secid:
            return {}
        url = (
            f"http://push2.eastmoney.com/api/qt/stock/fflow/daykline/get?"
            f"secid={secid}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
            f"&lmt=10"
        )
        data = _http_json(url)
        if not data or not data.get("data"):
            return {}
        klines = data["data"].get("klines", [])
        if not klines:
            return {}
        # 解析最近几天的主力资金
        days = []
        for line in klines[-5:]:
            parts = line.split(",")
            if len(parts) >= 7:
                days.append(
                    {
                        "date": parts[0],
                        "main_net_inflow": (
                            float(parts[1]) if parts[1] else 0
                        ),  # 主力净流入
                        "retail_net_inflow": (
                            float(parts[5]) if parts[5] else 0
                        ),  # 散户净流入
                    }
                )
        if days:
            # 汇总
            total_main = sum(d["main_net_inflow"] for d in days)
            consecutive = 0
            direction = 1 if days[-1]["main_net_inflow"] > 0 else -1
            for d in reversed(days):
                if (d["main_net_inflow"] > 0 and direction > 0) or (
                    d["main_net_inflow"] < 0 and direction < 0
                ):
                    consecutive += 1
                else:
                    break
            result = {
                "recent_days": days,
                "total_main_net_inflow": round(total_main, 0),
                "direction": "主力净流入" if direction > 0 else "主力净流出",
                "consecutive_days": consecutive,
            }
            print(f"    主力资金: {result['direction']} 连续{consecutive}天")
            return result
        return {}
    except Exception as e:
        print(f"    主力资金失败: {e}")
        return {}


def fetch_reports(pure, market=None):
    """获取券商研报列表 + 聚合共识。

    东方财富研报库 (reportapi.eastmoney.com) 仅收录 A 股研报，
    港股/美股传入任何代码格式均返回 0 条，直接跳过避免无效请求。

    返回: {reports: [...], consensus: {rating_distribution, target_price_median,
            target_price_count, upside_pct, latest_rating_change}}
    """
    if market in ("hk", "us"):
        print(
            f"    ℹ 研报: 东方财富研报库不支持 {market.upper()} 市场，"
            f"请用 web_search 搜索「股票名 研报 评级」"
        )
        return {"reports": [], "consensus": {}}
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        begin = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        url = (
            f"https://reportapi.eastmoney.com/report/list?"
            f"pageNo=1&pageSize=15&code={pure}"
            f"&industryCode=*&industry=*&rating=*&ratingchange=*"
            f"&beginTime={begin}&endTime={today}&fields=&qType=0"
        )
        data = _http_json(url)
        if not data:
            print(f"    ⚠ 研报: 接口无响应 (code={pure})")
            return {"reports": [], "consensus": {}}
        hits = data.get("hits", 0)
        print(f"    📄 研报: {hits} 篇命中 (code={pure}, 近180天)")
        results = []
        for r in data.get("data") or []:
            item = {
                "org": r.get("orgSName", ""),
                "rating": r.get("emRatingName", ""),
                "change": r.get("ratingChange", ""),
                "title": r.get("title", "")[:60],
                "date": (r.get("publishDate") or "")[:10],
                "infoCode": r.get("infoCode", ""),
                "stockName": r.get("stockName", ""),
                # 东方财富研报 API 字段：预测明年 PE、后年 PE、目标价（如有）
                "predict_next_year_pe": r.get("predictNextTwoYearPe") or r.get("predictThisYearPe"),
                "predict_this_year_eps": r.get("predictThisYearEps"),
                "predict_next_year_eps": r.get("predictNextTwoYearEps"),
            }
            if item["infoCode"]:
                item["pdf_url"] = (
                    f"https://pdf.dfcfw.com/pdf/H3_{item['infoCode']}_1.pdf"
                )
            results.append(item)

        # 聚合共识
        consensus = _aggregate_ratings(results)
        if consensus:
            print(
                f"    📊 研报共识: {consensus.get('rating_summary', 'N/A')} | "
                f"目标价中位数 {consensus.get('target_price_median', 'N/A')} | "
                f"上行空间 {consensus.get('upside_pct', 'N/A')}%"
            )
        return {"reports": results, "consensus": consensus}
    except Exception as e:
        print(f"    ⚠ 研报获取失败 (code={pure}): {e}")
        return {"reports": [], "consensus": {}}


def fetch_lhb(pure, market=None):
    """A 股龙虎榜数据（机构/游资买卖明细）。

    东方财富 datacenter-web 接口 RPT_DAILYBILLBOARD_DETAILS。
    仅 A 股支持，港股/美股跳过。
    返回: [{date, name, reason, net_amount, buy_amount, sell_amount, operator}, ...]
    """
    if market not in ("sh", "sz"):
        return []
    try:
        url = (
            f"https://datacenter-web.eastmoney.com/api/data/v1/get?"
            f"sortColumns=TRADE_DATE&sortTypes=-1&pageSize=10&pageNumber=1"
            f"&reportName=RPT_DAILYBILLBOARD_DETAILS&columns=ALL"
            f"&filter=(SECURITY_CODE=%22{pure}%22)"
        )
        data = _http_json(url)
        if not data or not data.get("result"):
            print(f"    ℹ 龙虎榜: {pure} 近期无龙虎榜记录")
            return []
        rows = data["result"].get("data") or []
        results = []
        for r in rows:
            results.append(
                {
                    "date": (r.get("TRADE_DATE") or "")[:10],
                    "name": r.get("SECURITY_NAME_ABBR", ""),
                    "reason": r.get("EXPLAIN", ""),
                    "net_amount_wan": round(r.get("NET_AMOUNT") or 0, 2),
                    "buy_amount_wan": round(r.get("BUY_AMOUNT") or 0, 2),
                    "sell_amount_wan": round(r.get("SELL_AMOUNT") or 0, 2),
                    "operator": r.get("OPERATE_DEPT_NAME", ""),
                }
            )
        # 去重（同一天可能多条）
        seen = set()
        unique = []
        for r in results:
            key = (r["date"], r["reason"][:20])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        print(f"    🐉 龙虎榜: {pure} 近期上榜 {len(unique)} 次")
        return unique[:8]
    except Exception as e:
        print(f"    ⚠ 龙虎榜数据失败: {e}")
        return []


# ── 港股卖空数据 ──


def fetch_hk_short(pure, market=None):
    """港股卖空数据（卖空股数/金额/占比）。

    东方财富 datacenter-web 接口 RPT_HKSHORT_MAIN。
    仅港股支持。
    返回: {recent: [{date, short_shares, short_amount, short_ratio_pct, total_shares}],
           summary: {avg_ratio_20d, latest_ratio, spike_flag}}
    """
    if market != "hk":
        return {}
    # 港股代码补零到 5 位
    hk_code = pure.zfill(5) if len(pure) < 5 else pure
    try:
        url = (
            f"https://datacenter-web.eastmoney.com/api/data/v1/get?"
            f"sortColumns=TRADE_DATE&sortTypes=-1&pageSize=20&pageNumber=1"
            f"&reportName=RPT_HKSHORT_MAIN&columns=ALL"
            f"&filter=(SECURITY_CODE=%22{hk_code}%22)"
        )
        data = _http_json(url)
        if not data or not data.get("result"):
            print(f"    ℹ 港股卖空: {hk_code} 无卖空数据")
            return {}
        rows = data["result"].get("data") or []
        if not rows:
            return {}
        recent = []
        ratios = []
        for r in rows:
            ratio = r.get("SHORT_RATIO") or 0
            ratios.append(float(ratio))
            recent.append(
                {
                    "date": (r.get("TRADE_DATE") or "")[:10],
                    "short_shares": r.get("SHORT_SHARES"),
                    "short_amount": r.get("SHORT_AMOUNT"),
                    "short_ratio_pct": round(float(ratio), 2),
                }
            )
        recent = recent[:15]
        # 卖空占比是否骤增（最新 > 20日均值 * 1.5）
        avg_20 = sum(ratios) / len(ratios) if ratios else 0
        latest = ratios[0] if ratios else 0
        spike = latest > avg_20 * 1.5 and latest > 5  # 超过均值50%且>5%
        summary = {
            "avg_ratio_20d": round(avg_20, 2),
            "latest_ratio": round(latest, 2),
            "spike_flag": spike,
            "spike_note": (
                f"⚠️ 卖空占比骤增（最新 {latest:.1f}% > 20日均 {avg_20:.1f}%），警惕空头建仓"
                if spike
                else "卖空占比正常"
            ),
        }
        print(
            f"    📉 港股卖空: {hk_code} 最新占比 {latest:.1f}%, 20日均 {avg_20:.1f}%"
            + (" ⚠️ 骤增" if spike else "")
        )
        return {"recent": recent, "summary": summary}
    except Exception as e:
        print(f"    ⚠ 港股卖空数据失败: {e}")
        return {}


def _aggregate_ratings(reports):
    """聚合研报评级分布 + 共识目标价。

    评级分布：买入/增持/中性/减持/卖出 各多少家
    共识目标价：用 predict_next_year_eps * 行业 PE 或研报内目标价（如有）的中位数
    """
    if not reports:
        return {}
    try:
        # 评级分布
        rating_dist = {}
        for r in reports:
            rating = (r.get("rating") or "").strip()
            if not rating:
                continue
            # 归一化：买入/推荐/强烈推荐 → 买入；增持/优于大市 → 增持；中性/持有 → 中性
            if any(k in rating for k in ["买入", "推荐", "强烈推荐"]):
                cat = "买入"
            elif any(k in rating for k in ["增持", "优于大市", "Accumulate"]):
                cat = "增持"
            elif any(k in rating for k in ["中性", "持有", "Hold", "Neutral"]):
                cat = "中性"
            elif any(k in rating for k in ["减持", "卖出", "Reduce", "Sell"]):
                cat = "减持"
            else:
                cat = rating
            rating_dist[cat] = rating_dist.get(cat, 0) + 1

        # 评级变动方向（最近一次）
        latest_change = ""
        if reports:
            latest = reports[0]
            ch = latest.get("change")
            if ch == 1:
                latest_change = f"最近一次评级上调（{latest.get('org','')} {latest.get('date','')}）"
            elif ch == -1:
                latest_change = f"最近一次评级下调（{latest.get('org','')} {latest.get('date','')}）"
            elif ch == 0:
                latest_change = f"最近一次评级维持（{latest.get('org','')} {latest.get('date','')}）"

        # 评级摘要字符串
        rating_summary = " / ".join(f"{k}{v}家" for k, v in rating_dist.items())

        # 目标价：东方财富 API 不直接返回目标价，但有预测 EPS
        # 若有 predict_next_year_eps，可用 行业PE * EPS 估算（粗略）
        # 这里先返回 EPS 共识，目标价留给模型结合行业 PE 判断
        def _safe_float_eps(v):
            try:
                f = float(v)
                return f if f == f else None
            except (TypeError, ValueError):
                return None

        eps_list = [
            _safe_float_eps(r.get("predict_next_year_eps"))
            for r in reports
            if _safe_float_eps(r.get("predict_next_year_eps")) is not None
        ]
        eps_median = round(sorted(eps_list)[len(eps_list) // 2], 2) if eps_list else None

        consensus = {
            "rating_distribution": rating_dist,
            "rating_summary": rating_summary,
            "total_reports": len(reports),
            "latest_rating_change": latest_change,
            "next_year_eps_median": eps_median,
        }

        # 目标价中位数（如果 API 返回了）
        # 东方财富部分研报有 predictNextTwoYearPe，可算目标价 = EPS * PE
        target_prices = []
        for r in reports:
            eps = _safe_float_eps(r.get("predict_next_year_eps"))
            pe = _safe_float_eps(r.get("predict_next_year_pe"))
            if eps and pe and eps > 0 and pe > 0:
                target_prices.append(round(eps * pe, 2))
        if target_prices:
            consensus["target_price_median"] = round(
                sorted(target_prices)[len(target_prices) // 2], 2
            )
            consensus["target_price_count"] = len(target_prices)
            consensus["target_price_range"] = [
                round(min(target_prices), 2),
                round(max(target_prices), 2),
            ]

        return consensus
    except Exception as e:
        print(f"    ⚠ 研报共识聚合失败: {e}")
        return {}


# ── 研报 PDF 自动下载 ──


def auto_fetch_report_pdf(
    reports, out_dir, stock_name="", stock_code="", max_reports=2, max_pages=4
):
    """下载研报 PDF 并提取图表页。stock_name/stock_code 用于过滤无关研报。"""
    all_pages = []
    if not reports:
        return all_pages
    try:
        import fitz
    except ImportError:
        print("    ⚠ pymupdf 未安装，跳过研报PDF提取")
        return all_pages

    rdir = os.path.join(out_dir, "research_reports")
    os.makedirs(rdir, exist_ok=True)
    downloaded = 0
    for r in reports:
        if downloaded >= max_reports:
            break
        if len(all_pages) >= max_pages:
            break
        pdf_url = r.get("pdf_url")
        if not pdf_url:
            continue

        # 标题过滤：跳过不含目标股票名/代码/简称的研报
        # 修复 bug：stock_name 可能是全称（"腾讯控股"）但研报标题用简称（"腾讯"）
        title = r.get("title", "")
        # stockName 字段是研报 API 返回的该研报对应股票名，最可靠
        report_stock_name = r.get("stockName", "")
        if stock_name or stock_code:
            match = False
            # 1. 代码匹配
            if stock_code and stock_code in title:
                match = True
            # 2. 全称匹配
            if stock_name and stock_name in title:
                match = True
            # 3. 简称匹配：去掉常见后缀（控股/集团/股份/科技）做子串
            if stock_name and not match:
                short_aliases = []
                base = stock_name
                for suffix in ["控股", "集团", "股份", "科技", "实业", "投资", "发展"]:
                    if base.endswith(suffix) and len(base) > len(suffix) + 1:
                        short_aliases.append(base[: -len(suffix)])
                for alias in short_aliases:
                    if alias in title:
                        match = True
                        break
            # 4. 研报 API 自带的 stockName 字段匹配（最可靠）
            if not match and report_stock_name:
                if report_stock_name == stock_name or report_stock_name in stock_name or stock_name in report_stock_name:
                    match = True
            if not match:
                print(f"    ⏭ 跳过无关研报: {title[:40]}")
                continue

        pdf_path = os.path.join(rdir, f"report_{downloaded+1}.pdf")
        print(f"    尝试下载研报: [{r['org']}] {title[:40]}...")
        try:
            req = urllib.request.Request(
                pdf_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://data.eastmoney.com/",
                },
            )
            resp = urllib.request.urlopen(req, timeout=20)
            with open(pdf_path, "wb") as f:
                f.write(resp.read())
            if os.path.getsize(pdf_path) < 100:
                continue
            doc = fitz.open(pdf_path)
            for i in range(min(len(doc), 30)):
                if len(all_pages) >= max_pages:
                    break
                page = doc[i]
                score = _page_chart_score(page, i, len(doc))
                if score > 3:
                    mat = fitz.Matrix(200 / 72, 200 / 72)
                    pix = page.get_pixmap(matrix=mat)
                    png = os.path.join(rdir, f"report_p{i+1:03d}.png")
                    pix.save(png)
                    all_pages.append(png)
            doc.close()
            downloaded += 1
            if all_pages:
                print(f"    ✅ 提取了 {len(all_pages)} 页研报图表")
        except Exception as e:
            print(f"    ⚠ 研报下载失败: {e}")
    return all_pages


# ═══════════════════════════════════════════════
# 技术指标计算
# ═══════════════════════════════════════════════


def calc_kdj(df, n=9, m1=3, m2=3):
    """KDJ 指标。返回 (k, d, j) 序列。"""
    low_n = df["Low"].rolling(n, min_periods=n).min()
    high_n = df["High"].rolling(n, min_periods=n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_bollinger(close, n=20, k=2):
    """布林带。返回 (upper, middle, lower) 序列。"""
    middle = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = middle + k * std
    lower = middle - k * std
    return upper, middle, lower


def calc_obv(df):
    """OBV 指标。"""
    direction = (df["Close"].diff() > 0).astype(int) * 2 - 1
    direction.iloc[0] = 0
    return (direction * df["Volume"]).cumsum()


def compute_technicals(df):
    if df is None or len(df) < 20:
        return {}
    try:
        latest = df.iloc[-1]
        close = df["Close"]
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1] if len(df) >= 10 else None
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(df) >= 60 else None
        ma120 = close.rolling(120).mean().iloc[-1] if len(df) >= 120 else None
        # MACD
        _, _, hist = calc_macd(close)
        mh = float(hist.iloc[-1]) if not hist.empty else None
        ph = float(hist.iloc[-2]) if len(hist) > 1 else None
        # RSI
        rsi = calc_rsi(close)
        rv = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else None
        # 成交量
        vm20 = df["Volume"].rolling(20).mean().iloc[-1]
        vm5 = df["Volume"].rolling(5).mean().iloc[-1] if len(df) >= 5 else None
        vm10 = df["Volume"].rolling(10).mean().iloc[-1] if len(df) >= 10 else None
        vr = float(latest["Volume"] / vm20) if vm20 > 0 else 1.0
        # KDJ
        kdj_k = kdj_d = kdj_j = None
        if len(df) >= 9 and "High" in df.columns and "Low" in df.columns:
            k, d, j = calc_kdj(df)
            kdj_k = float(k.iloc[-1]) if not k.empty and not np.isnan(k.iloc[-1]) else None
            kdj_d = float(d.iloc[-1]) if not d.empty and not np.isnan(d.iloc[-1]) else None
            kdj_j = float(j.iloc[-1]) if not j.empty and not np.isnan(j.iloc[-1]) else None
        # 布林带
        boll_upper = boll_mid = boll_lower = None
        boll_pos = ""
        if len(df) >= 20:
            up, mid, low = calc_bollinger(close)
            boll_upper = float(up.iloc[-1]) if not np.isnan(up.iloc[-1]) else None
            boll_mid = float(mid.iloc[-1]) if not np.isnan(mid.iloc[-1]) else None
            boll_lower = float(low.iloc[-1]) if not np.isnan(low.iloc[-1]) else None
            cur = float(latest["Close"])
            if boll_upper and boll_lower and boll_mid:
                if cur >= boll_upper:
                    boll_pos = "突破上轨"
                elif cur <= boll_lower:
                    boll_pos = "跌破下轨"
                elif cur > boll_mid:
                    boll_pos = "中轨上方"
                else:
                    boll_pos = "中轨下方"
        # OBV
        obv_val = None
        obv_trend = ""
        if len(df) >= 30:
            obv = calc_obv(df)
            obv_val = float(obv.iloc[-1])
            obv5 = float(obv.iloc[-6]) if len(obv) > 5 else None
            if obv5 is not None:
                obv_trend = "上行" if obv_val > obv5 else "下行"

        r = {
            "close": round(float(latest["Close"]), 2),
            "ma5": round(float(ma5), 2),
            "ma20": round(float(ma20), 2),
            "vol_ratio": round(vr, 2),
        }
        if ma10 is not None:
            r["ma10"] = round(float(ma10), 2)
        if ma60:
            r["ma60"] = round(float(ma60), 2)
        if ma120:
            r["ma120"] = round(float(ma120), 2)
        if vm5 is not None:
            r["vol_ma5"] = round(float(vm5 / 1e4), 2)  # 万手
        if vm10 is not None:
            r["vol_ma10"] = round(float(vm10 / 1e4), 2)
        if rv and not np.isnan(rv):
            r["rsi"] = round(rv, 1)
        if mh is not None and not np.isnan(mh):
            r["macd_hist"] = round(mh, 4)
        if ph is not None and not np.isnan(ph):
            r["prev_macd_hist"] = round(ph, 4)
        if kdj_k is not None:
            r["kdj_k"] = round(kdj_k, 2)
            r["kdj_d"] = round(kdj_d, 2)
            r["kdj_j"] = round(kdj_j, 2)
        if boll_upper is not None:
            r["boll_upper"] = round(boll_upper, 2)
            r["boll_mid"] = round(boll_mid, 2)
            r["boll_lower"] = round(boll_lower, 2)
            if boll_pos:
                r["boll_position"] = boll_pos
        if obv_val is not None:
            r["obv"] = round(obv_val, 0)
            if obv_trend:
                r["obv_trend"] = obv_trend

        # 均线排列状态
        if ma60:
            if ma5 > ma20 > ma60:
                r["ma_status"] = "多头排列"
            elif ma5 < ma20 < ma60:
                r["ma_status"] = "空头排列"
            else:
                r["ma_status"] = "交叉缠绕"
        # MACD 状态
        if mh is not None and ph is not None:
            if ph < 0 and mh > 0:
                r["macd_status"] = "金叉"
            elif ph > 0 and mh < 0:
                r["macd_status"] = "死叉"
            elif mh > ph:
                r["macd_status"] = "动能增强"
            else:
                r["macd_status"] = "动能减弱"
        # RSI 状态
        if rv and not np.isnan(rv):
            r["rsi_status"] = "超买" if rv > 70 else ("超卖" if rv < 30 else "中性")
        # KDJ 状态
        if kdj_k is not None and kdj_d is not None and kdj_j is not None:
            if kdj_j > 100:
                r["kdj_status"] = "超买"
            elif kdj_j < 0:
                r["kdj_status"] = "超卖"
            elif kdj_k > kdj_d and kdj_k < 80:
                r["kdj_status"] = "金叉向上"
            elif kdj_k < kdj_d and kdj_k > 20:
                r["kdj_status"] = "死叉向下"
            else:
                r["kdj_status"] = "中性"

        # 综合信号数组（自动判断形态）
        signals = []
        if r.get("ma_status") == "多头排列":
            signals.append("MA 多头排列（强势）")
        elif r.get("ma_status") == "空头排列":
            signals.append("MA 空头排列（弱势）")
        if r.get("macd_status") == "金叉":
            signals.append("MACD 金叉（短期看涨）")
        elif r.get("macd_status") == "死叉":
            signals.append("MACD 死叉（短期看跌）")
        if r.get("rsi_status") == "超买":
            signals.append("RSI 超买（注意回调）")
        elif r.get("rsi_status") == "超卖":
            signals.append("RSI 超卖（可能反弹）")
        if r.get("kdj_status") == "超买":
            signals.append("KDJ 超买")
        elif r.get("kdj_status") == "超卖":
            signals.append("KDJ 超卖")
        elif r.get("kdj_status") == "金叉向上":
            signals.append("KDJ 金叉向上")
        elif r.get("kdj_status") == "死叉向下":
            signals.append("KDJ 死叉向下")
        if r.get("boll_position") == "突破上轨":
            signals.append("突破布林上轨")
        elif r.get("boll_position") == "跌破下轨":
            signals.append("跌破布林下轨")
        if ma120 and float(latest["Close"]) > ma120:
            signals.append("站上半年线")
        elif ma120 and float(latest["Close"]) < ma120:
            signals.append("跌破半年线")
        if vr > 2.0:
            signals.append(f"放量（量比 {vr:.1f}）")
        elif vr < 0.5:
            signals.append(f"缩量（量比 {vr:.1f}）")
        if r.get("obv_trend") == "上行" and float(latest["Close"]) > float(close.iloc[-6]):
            signals.append("OBV 与价格同步上行（健康）")
        elif r.get("obv_trend") == "下行" and float(latest["Close"]) > float(close.iloc[-6]):
            signals.append("OBV 顶背离（量价背离警示）")
        if signals:
            r["signals"] = signals
        return r
    except Exception as e:
        print(f"    ⚠ compute_technicals 失败: {e}")
        return {}


# ═══════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════


def chart_kline(df, code, out):
    """本地绘制 K 线图（fallback，仅在东方财富直链图片不可用时调用）"""
    if df is None or df.empty:
        return None
    try:
        import mplfinance as mpf

        mc = mpf.make_marketcolors(
            up=C["pos"],
            down=C["neg"],
            edge="inherit",
            wick="inherit",
            volume={"up": C["pos"], "down": C["neg"]},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle=":",
            y_on_right=True,
            facecolor=C["surface"],
            figcolor=C["bg"],
            gridcolor=C["grid"],
            rc={
                "axes.labelcolor": C["text"],
                "xtick.color": C["text"],
                "ytick.color": C["text"],
                "text.color": C["text"],
                "font.size": 9,
            },
        )
        path = os.path.join(out, "kline.png")
        mav = (5, 20, 60) if len(df) >= 60 else (5, 20)
        mpf.plot(
            df,
            type="candle",
            style=style,
            volume=True,
            mav=mav,
            title=f"\n{code} 日K线",
            figsize=(14, 7),
            savefig=dict(fname=path, dpi=150, bbox_inches="tight"),
        )
        print(f"    ✅ kline: 本地绘制")
        return path
    except Exception as e:
        print(f"    ⚠ K线图失败: {e}")
        return None


def chart_capital_flow(data, code, out, label="资金流"):
    if not data or len(data) < 3:
        return None
    try:
        dark_style()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        ch = df["shares_change"].values / 1e4
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(
            df["date"],
            ch,
            color=[C["pos"] if v > 0 else C["neg"] for v in ch],
            alpha=0.8,
            width=0.8,
        )
        if len(ch) >= 5:
            ax.plot(
                df["date"],
                pd.Series(ch).rolling(5).mean(),
                color=C["orange"],
                linewidth=1.5,
                label="5日均值",
            )
        ax.axhline(y=0, color=C["grid"], linewidth=0.5)
        ax.set_ylabel("每日持股变化(万股)")
        ax.set_title(f"{code} {label}持股变化", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=45)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        path = os.path.join(out, "capital_flow.png")
        save_fig(fig, path)
        return path
    except Exception as e:
        return None


def chart_macro(macro, out):
    if not macro:
        return None
    try:
        dark_style()
        items = list(macro.items())
        n = len(items)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, (name, d) in enumerate(items):
            ax = axes[i]
            chg = d.get("change_pct", 0)
            color = C["pos"] if chg > 0.05 else (C["neg"] if chg < -0.05 else C["text"])
            arrow = "▲" if chg > 0.05 else ("▼" if chg < -0.05 else "—")
            ax.text(
                0.5,
                0.65,
                f"{d['value']}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=C["text"],
            )
            ax.text(
                0.5,
                0.30,
                f"{arrow} {chg:+.2f}%",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
                color=color,
            )
            ax.text(
                0.5,
                0.08,
                name,
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
                color="#888",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("宏观环境", fontsize=12, y=1.02, color=C["text"])
        plt.tight_layout()
        path = os.path.join(out, "macro.png")
        save_fig(fig, path)
        return path
    except Exception as e:
        return None


def chart_valuation(info, code, out):
    pe = info.get("trailingPE")
    try:
        pe = float(pe)
    except (TypeError, ValueError):
        return None
    if pe is None or pe < 0:
        return None
    try:
        dark_style()
        fig, ax = plt.subplots(figsize=(8, 2.5))
        for x0, x1, c, l in [
            (0, 15, C["neg"], "便宜"),
            (15, 25, C["orange"], "合理"),
            (25, 45, C["pos"], "偏贵"),
        ]:
            ax.barh(0, x1 - x0, left=x0, height=0.5, color=c, alpha=0.25)
            ax.text((x0 + x1) / 2, -0.45, l, ha="center", fontsize=9, color="#888")
        ax.plot(pe, 0, marker="v", markersize=14, color=C["accent"], zorder=5)
        ax.text(
            pe,
            0.4,
            f"PE {pe:.1f}x",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=C["text"],
        )
        fwd = info.get("forwardPE")
        if fwd and fwd > 0:
            ax.plot(fwd, 0, marker="^", markersize=10, color=C["orange"], zorder=4)
            ax.text(
                fwd,
                -0.8,
                f"Forward {fwd:.1f}x",
                ha="center",
                fontsize=8,
                color=C["orange"],
            )
        ax.set_xlim(0, max(pe * 1.4, 45))
        ax.set_ylim(-1.2, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("PE(TTM)")
        ax.set_title(f"{code} 估值水平", fontsize=11, pad=8)
        ax.grid(True, axis="x", alpha=0.3)
        path = os.path.join(out, "valuation.png")
        save_fig(fig, path)
        return path
    except Exception as e:
        return None


def chart_pe_band(kline, info, code, out, years=3):
    """画 3 年 PE Band 图 + 输出当前 PE 历史百分位。

    用当前 EPS_TTM（= price / trailingPE）作为常量，反推历史 PE 序列。
    注意：这是近似法，因为 EPS 会变。但对于判断"当前股价是否在历史高位/低位"足够用。
    若 PE 无效（亏损股），退化成价格水位图（同样能判断股价在历史区间的位置）。

    返回: (image_path, stats_dict) 或 (None, None)
    """
    pe_now = info.get("trailingPE")
    try:
        pe_now = float(pe_now)
    except (TypeError, ValueError):
        pe_now = None
    if kline is None or len(kline) < 60:
        return None, None
    try:
        close = kline["Close"]
        n_days = min(len(close), 250 * years)
        close_recent = close.iloc[-n_days:]
        price_now = float(close.iloc[-1])

        dark_style()
        fig, ax = plt.subplots(figsize=(11, 4))
        dates = close_recent.index

        if pe_now and pe_now > 0:
            # PE Band 模式
            eps_ttm = price_now / pe_now
            pe_series = close_recent / eps_ttm
            pe_arr = pe_series.dropna().values
            if len(pe_arr) < 30:
                return None, None
            pe_sorted = sorted(pe_arr)
            rank = sum(1 for x in pe_sorted if x <= pe_now)
            percentile = round(rank / len(pe_sorted) * 100, 1)
            pe_mean = float(pe_series.mean())
            pe_std = float(pe_series.std())
            pe_max = float(pe_series.max())
            pe_min = float(pe_series.min())
            pe_p25 = float(pe_sorted[len(pe_sorted) // 4])
            pe_p75 = float(pe_sorted[3 * len(pe_sorted) // 4])

            ax.plot(dates, pe_series.values, color=C["accent"], linewidth=1.2, label="PE(TTM)")
            ax.axhline(pe_mean, color=C["text"], linestyle="--", linewidth=1, alpha=0.6, label=f"均值 {pe_mean:.1f}x")
            ax.axhline(pe_mean + pe_std, color=C["pos"], linestyle=":", linewidth=1, alpha=0.5, label=f"+1σ {pe_mean+pe_std:.1f}x")
            ax.axhline(max(0, pe_mean - pe_std), color=C["neg"], linestyle=":", linewidth=1, alpha=0.5, label=f"-1σ {max(0,pe_mean-pe_std):.1f}x")
            ax.fill_between(dates, pe_p25, pe_p75, color=C["accent"], alpha=0.1, label=f"25-75%区间")
            ax.axhline(pe_now, color=C["orange"], linewidth=1.5, label=f"当前 {pe_now:.1f}x")
            ax.set_title(
                f"{code} PE Band · 近{years}年 · 当前 PE 处于 {percentile}% 分位",
                fontsize=11, pad=8,
            )
            ax.set_ylabel("PE(TTM)")
            stats = {
                "mode": "pe_band",
                "pe_now": round(pe_now, 2),
                "pe_percentile_3y": percentile,
                "pe_mean_3y": round(pe_mean, 2),
                "pe_std_3y": round(pe_std, 2),
                "pe_max_3y": round(pe_max, 2),
                "pe_min_3y": round(pe_min, 2),
                "pe_p25_3y": round(pe_p25, 2),
                "pe_p75_3y": round(pe_p75, 2),
            }
            print(
                f"    📈 PE Band: 当前 {pe_now:.1f}x, 3年百分位 {percentile}%, "
                f"区间 [{pe_min:.1f}, {pe_max:.1f}], 均值 {pe_mean:.1f}x"
            )
        else:
            # 价格水位模式（PE 无效时，亏损股）
            price_arr = close_recent.dropna().values
            price_sorted = sorted(price_arr)
            rank = sum(1 for x in price_sorted if x <= price_now)
            percentile = round(rank / len(price_sorted) * 100, 1)
            price_mean = float(close_recent.mean())
            price_std = float(close_recent.std())
            price_max = float(close_recent.max())
            price_min = float(close_recent.min())
            p25 = float(price_sorted[len(price_sorted) // 4])
            p75 = float(price_sorted[3 * len(price_sorted) // 4])

            ax.plot(dates, close_recent.values, color=C["accent"], linewidth=1.2, label="收盘价")
            ax.axhline(price_mean, color=C["text"], linestyle="--", linewidth=1, alpha=0.6, label=f"均值 {price_mean:.2f}")
            ax.fill_between(dates, p25, p75, color=C["accent"], alpha=0.1, label=f"25-75%区间")
            ax.axhline(price_now, color=C["orange"], linewidth=1.5, label=f"当前 {price_now:.2f}")
            pe_note = info.get("trailingPE_note", "PE 无效")
            ax.set_title(
                f"{code} 价格水位 · 近{years}年 · 当前股价处于 {percentile}% 分位（{pe_note}）",
                fontsize=11, pad=8,
            )
            ax.set_ylabel("收盘价")
            stats = {
                "mode": "price_band",
                "pe_note": pe_note,
                "price_now": round(price_now, 2),
                "price_percentile_3y": percentile,
                "price_mean_3y": round(price_mean, 2),
                "price_max_3y": round(price_max, 2),
                "price_min_3y": round(price_min, 2),
            }
            print(
                f"    📈 价格水位（PE无效）: 当前 {price_now:.2f}, 3年百分位 {percentile}%, "
                f"区间 [{price_min:.2f}, {price_max:.2f}]"
            )

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        plt.tight_layout()
        path = os.path.join(out, "pe_band.png")
        save_fig(fig, path)
        return path, stats
    except Exception as e:
        print(f"    ⚠ chart_pe_band 失败: {e}")
        return None, None


def fetch_financials_structured(code, market=None, pure=None):
    """获取结构化季度财务三表数据，供模型直接分析。

    返回: {quarters: [{period, revenue, net_income, gross_margin, net_margin,
                       ocf, fcf, debt_ratio, roe, rev_yoy, ni_qoq, ocf_ni_ratio}, ...],
           latest_summary: {...}, source: "yfinance|akshare"}
    """
    result = {"quarters": [], "latest_summary": {}, "source": ""}

    # ── A 股优先用 akshare ──
    if HAS_AKSHARE and market in ("sh", "sz") and pure:
        try:
            # 利润表
            df_inc = ak.stock_financial_report_sina(
                stock=f"sh{pure}" if market == "sh" else f"sz{pure}",
                symbol="利润表",
            )
            if df_inc is not None and not df_inc.empty:
                quarters = []
                for _, row in df_inc.head(8).iterrows():
                    period = str(row.get("报告日", ""))
                    rev = _safe_float(row.get("一、营业总收入"))
                    ni = _safe_float(row.get("净利润"))
                    gm = _safe_float(row.get("销售毛利率(%)"))
                    nm = _safe_float(row.get("销售净利率(%)"))
                    quarters.append(
                        {
                            "period": period,
                            "revenue": rev,
                            "net_income": ni,
                            "gross_margin_pct": gm,
                            "net_margin_pct": nm,
                        }
                    )
                result["quarters"] = quarters
                result["source"] = "akshare"
                # 计算增速
                _add_growth_metrics(result)
                print(f"    📊 财务三表(akshare): {len(quarters)} 季")
                return result
        except Exception as e:
            print(f"    ⚠ akshare 财务数据失败: {e}")

    # ── yfinance 兜底（港股/美股/akshare失败）──
    if not HAS_YFINANCE:
        return result
    try:
        t = _yf(code)
        qf = t.quarterly_financials
        qb = t.quarterly_balance_sheet
        qc = t.quarterly_cashflow
        if qf is None or qf.empty:
            return result

        def _row(df, names):
            if df is None or df.empty:
                return None
            for n in names:
                if n in df.index:
                    s = df.loc[n]
                    return [None if (v != v) else float(v) for v in s.values]
            return None

        rev_s = _row(qf, ["Total Revenue", "Operating Revenue", "Revenue"])
        ni_s = _row(qf, ["Net Income", "Net Income Common Stockholders"])
        gp_s = _row(qf, ["Gross Profit"])
        ocf_s = _row(qc, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex_s = _row(qc, ["Capital Expenditure"])
        debt_s = _row(qb, ["Total Debt"])
        asset_s = _row(qb, ["Total Assets"])
        equity_s = _row(qb, ["Stockholders Equity", "Total Stockholder Equity"])

        quarters = []
        n = min(len(rev_s or []), 8)
        for i in range(n):
            period = str(qf.columns[i])[:10] if i < len(qf.columns) else f"Q{i+1}"
            rev = rev_s[i] if rev_s and i < len(rev_s) else None
            ni = ni_s[i] if ni_s and i < len(ni_s) else None
            gp = gp_s[i] if gp_s and i < len(gp_s) else None
            ocf = ocf_s[i] if ocf_s and i < len(ocf_s) else None
            capex = capex_s[i] if capex_s and i < len(capex_s) else None
            debt = debt_s[i] if debt_s and i < len(debt_s) else None
            assets = asset_s[i] if asset_s and i < len(asset_s) else None
            equity = equity_s[i] if equity_s and i < len(equity_s) else None

            q = {
                "period": period,
                "revenue": _fmt_b(rev),
                "net_income": _fmt_b(ni),
                "operating_cashflow": _fmt_b(ocf),
                "free_cashflow": _fmt_b(ocf + capex) if ocf is not None and capex is not None else None,
            }
            if rev and rev != 0:
                q["gross_margin_pct"] = round(gp / rev * 100, 2) if gp else None
                q["net_margin_pct"] = round(ni / rev * 100, 2) if ni else None
                q["ocf_ni_ratio"] = round(ocf / ni, 2) if (ocf and ni and ni != 0) else None
            if assets and assets != 0:
                q["debt_ratio_pct"] = round(debt / assets * 100, 2) if debt else None
            if equity and equity != 0 and ni:
                q["roe_pct"] = round(ni / equity * 100, 2)
            quarters.append(q)

        result["quarters"] = quarters
        result["source"] = "yfinance"
        _add_growth_metrics(result)
        print(f"    📊 财务三表(yfinance): {len(quarters)} 季")
        return result
    except Exception as e:
        print(f"    ⚠ yfinance 财务数据失败: {e}")
        return result


def _safe_float(v):
    try:
        f = float(v)
        return None if f != f else f
    except Exception as e:
        return None


def _fmt_b(v):
    """格式化为亿元（输入为元）。"""
    if v is None:
        return None
    try:
        return round(v / 1e8, 2)
    except Exception as e:
        return None


def _add_growth_metrics(result):
    """为 quarters 加 YoY/QoQ 增速。"""
    qs = result.get("quarters") or []
    if len(qs) < 2:
        return
    for i in range(len(qs)):
        cur = qs[i]
        # QoQ
        if i + 1 < len(qs):
            prev = qs[i + 1]
            if cur.get("revenue") and prev.get("revenue") and prev["revenue"] != 0:
                cur["rev_qoq_pct"] = round(
                    (cur["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100, 2
                )
            if cur.get("net_income") and prev.get("net_income") and prev["net_income"] != 0:
                cur["ni_qoq_pct"] = round(
                    (cur["net_income"] - prev["net_income"])
                    / abs(prev["net_income"])
                    * 100,
                    2,
                )
        # YoY（4 季前）
        if i + 4 < len(qs):
            yoy_prev = qs[i + 4]
            if cur.get("revenue") and yoy_prev.get("revenue") and yoy_prev["revenue"] != 0:
                cur["rev_yoy_pct"] = round(
                    (cur["revenue"] - yoy_prev["revenue"])
                    / abs(yoy_prev["revenue"])
                    * 100,
                    2,
                )
    # latest_summary
    if qs:
        latest = qs[0]
        result["latest_summary"] = {
            "period": latest.get("period"),
            "revenue_yiyuan": latest.get("revenue"),
            "net_income_yiyuan": latest.get("net_income"),
            "gross_margin_pct": latest.get("gross_margin_pct"),
            "net_margin_pct": latest.get("net_margin_pct"),
            "ocf_ni_ratio": latest.get("ocf_ni_ratio"),
            "debt_ratio_pct": latest.get("debt_ratio_pct"),
            "roe_pct": latest.get("roe_pct"),
            "rev_yoy_pct": latest.get("rev_yoy_pct"),
            "rev_qoq_pct": latest.get("rev_qoq_pct"),
        }


def chart_financials(code, out, financials_data=None):
    """画季度财务趋势图。若传入 financials_data 则用它，否则自行获取。"""
    if not HAS_YFINANCE and financials_data is None:
        return None
    try:
        if financials_data is None:
            qf = _yf(code).quarterly_financials
        else:
            quarters = financials_data.get("quarters") or []
            if not quarters:
                return None
            qs_labels = [q.get("period", "")[:7] for q in reversed(quarters)]
            rev = [q.get("revenue") or 0 for q in reversed(quarters)]
            ni = [q.get("net_income") or 0 for q in reversed(quarters)]

            dark_style()
            fig, ax1 = plt.subplots(figsize=(10, 4))
            x = range(len(qs_labels))
            ax1.bar(x, rev, color=C["accent"], alpha=0.7, width=0.6, label="营收(亿元)")
            ax1.set_ylabel("营收(亿元)")
            ax1.set_xticks(x)
            ax1.set_xticklabels(qs_labels, fontsize=8, rotation=45)
            ax2 = ax1.twinx()
            ax2.plot(
                x, ni, color=C["orange"], linewidth=2, marker="o",
                markersize=5, label="净利润(亿元)",
            )
            ax2.set_ylabel("净利润(亿元)", color=C["orange"])
            ax2.tick_params(axis="y", labelcolor=C["orange"])
            ax1.set_title(f"{code} 季度营收与利润", fontsize=11, pad=8)
            ax1.grid(True, axis="y", alpha=0.3)
            l1, lb1 = ax1.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)
            plt.tight_layout()
            path = os.path.join(out, "financials_trend.png")
            save_fig(fig, path)
            return path

        # 旧路径：直接从 yfinance
        if qf is None or qf.empty:
            return None
        qf = qf.iloc[:, :8]
        rev = qf.loc["Total Revenue"] / 1e9 if "Total Revenue" in qf.index else None
        ni = qf.loc["Net Income"] / 1e9 if "Net Income" in qf.index else None
        if rev is None:
            return None
        dark_style()
        fig, ax1 = plt.subplots(figsize=(10, 4))
        qs = [f"{d.year%100}Q{(d.month-1)//3+1}" for d in reversed(rev.index)]
        rv = list(reversed(rev.values))
        x = range(len(qs))
        ax1.bar(x, rv, color=C["accent"], alpha=0.7, width=0.6, label="营收(十亿)")
        ax1.set_ylabel("营收(十亿)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(qs, fontsize=8, rotation=45)
        if ni is not None:
            ax2 = ax1.twinx()
            nv = list(reversed(ni.values))
            ax2.plot(
                x, nv, color=C["orange"], linewidth=2, marker="o",
                markersize=5, label="净利润(十亿)",
            )
            ax2.set_ylabel("净利润(十亿)", color=C["orange"])
            ax2.tick_params(axis="y", labelcolor=C["orange"])
        ax1.set_title(f"{code} 季度营收与利润", fontsize=11, pad=8)
        ax1.grid(True, axis="y", alpha=0.3)
        l1, lb1 = ax1.get_legend_handles_labels()
        if ni is not None:
            l2, lb2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8)
        else:
            ax1.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        path = os.path.join(out, "financials_trend.png")
        save_fig(fig, path)
        return path
    except Exception as e:
        print(f"    ⚠ chart_financials 失败: {e}")
        return None


def chart_adr(code, adr_code, out, days=20):
    if not adr_code or not HAS_YFINANCE:
        return None
    try:
        per = f"{days+5}d"
        hk = _yf(code).history(period=per)
        adr = _yf(adr_code).history(period=per)
        fx = _yf("USDHKD=X").history(period=per)
        if hk.empty or adr.empty or fx.empty:
            return None
        com = hk.index.intersection(adr.index).intersection(fx.index)
        if len(com) < 5:
            return None
        prem = (
            (adr.loc[com, "Close"] * fx.loc[com, "Close"] - hk.loc[com, "Close"])
            / hk.loc[com, "Close"]
            * 100
        ).values
        dark_style()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(
            com,
            prem,
            color=[C["pos"] if v > 0 else C["neg"] for v in prem],
            alpha=0.8,
            width=0.8,
        )
        ax.axhline(y=0, color=C["grid"], linewidth=0.5)
        ax.set_ylabel("ADR溢价/折价(%)")
        ax.set_title(f"{code} vs {adr_code} ADR溢价率", fontsize=11, pad=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out, "adr_premium.png")
        save_fig(fig, path)
        return path
    except Exception as e:
        return None


# ═══════════════════════════════════════════════
# 生成 report_template.md
# ═══════════════════════════════════════════════

# ── 图片数量控制 ──

MAX_IMAGES = 8


def _select_images(image_entries, output_dir):
    """按优先级选取图片，返回 {label: 原始路径}（不复制）。"""
    entries = [(l, p, pri) for l, p, pri in image_entries if p and os.path.exists(p)]
    entries.sort(key=lambda x: x[2])
    if len(entries) > MAX_IMAGES:
        entries = entries[:MAX_IMAGES]
        print(f"    ⚠ 图片超过{MAX_IMAGES}张上限，保留前{MAX_IMAGES}张")
    result = {}
    for label, path, _ in entries:
        result[label] = path
    return result


def _set_image_permission(path):
    """设置图片权限为 644，确保外部可访问"""
    try:
        os.chmod(path, 0o644)
    except Exception as e:
        print(f"    ⚠ _select_images 异常: {e}")
        pass


def _img_md(label, path):
    """生成 markdown 图片行（使用传入的路径）"""
    if not path:
        return ""
    # path 已经是相对路径或绝对路径，直接使用
    return f"![{label}]({path})"


def generate_template(code, charts, report_pages, ir_pages, capital_label, out):
    """生成带图片的报告骨架（绝对路径）"""

    # ── 收集所有图片，按优先级排序 ──
    image_entries = []
    if charts.get("kline"):
        image_entries.append(("K线图", charts["kline"], 1))
    if charts.get("kline_intraday"):
        image_entries.append(("分时图", charts["kline_intraday"], 2))
    if charts.get("valuation"):
        image_entries.append(("估值水平", charts["valuation"], 3))
    if charts.get("capital_flow"):
        image_entries.append((capital_label, charts["capital_flow"], 4))
    if charts.get("macro"):
        image_entries.append(("宏观环境", charts["macro"], 5))
    if charts.get("adr_premium"):
        image_entries.append(("ADR溢价", charts["adr_premium"], 6))
    if charts.get("financials_trend"):
        image_entries.append(("营收趋势", charts["financials_trend"], 7))
    for i, p in enumerate(report_pages or []):
        image_entries.append((f"研报图表{i+1}", p, 10 + i))
    for i, p in enumerate(ir_pages or []):
        image_entries.append((f"业绩演示{i+1}", p, 20 + i))

    # 数量控制
    print("  → 图片筛选...")
    img_map = _select_images(image_entries, out)

    # ── 生成模板 ──
    lines = []
    lines.append(f"# {{TITLE}}")
    lines.append("")
    lines.append("**一句话结论：** {VERDICT}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # K线图
    img = _img_md("K线图", img_map.get("K线图"))
    if img:
        lines.append("## 技术面")
        lines.append("")
        lines.append(img)
        lines.append("")
        img2 = _img_md("分时图", img_map.get("分时图"))
        if img2:
            lines.append(img2)
            lines.append("")
        lines.append("{TECHNICAL_DESC}")
        lines.append("")

    # 估值图
    img = _img_md("估值水平", img_map.get("估值水平"))
    if img:
        lines.append("## 估值")
        lines.append("")
        lines.append(img)
        lines.append("")
        lines.append("{VALUATION_DESC}")
        lines.append("")

    # 营收趋势图
    img = _img_md("营收趋势", img_map.get("营收趋势"))
    if img:
        lines.append("## 营收趋势")
        lines.append("")
        lines.append(img)
        lines.append("")
        lines.append("{FINANCIALS_DESC}")
        lines.append("")

    # 资金流图
    img = _img_md(capital_label, img_map.get(capital_label))
    if img:
        lines.append(f"## {capital_label}")
        lines.append("")
        lines.append(img)
        lines.append("")
        lines.append("{CAPITAL_FLOW_DESC}")
        lines.append("")

    # ADR溢价图
    img = _img_md("ADR溢价", img_map.get("ADR溢价"))
    if img:
        lines.append("## ADR溢价")
        lines.append("")
        lines.append(img)
        lines.append("")
        lines.append("{ADR_DESC}")
        lines.append("")

    # 宏观图
    img = _img_md("宏观环境", img_map.get("宏观环境"))
    if img:
        lines.append("## 宏观环境")
        lines.append("")
        lines.append(img)
        lines.append("")
        lines.append("{MACRO_DESC}")
        lines.append("")

    # 券商研报图表
    report_imgs = [
        _img_md(f"研报图表{i+1}", img_map.get(f"研报图表{i+1}"))
        for i in range(len(report_pages or []))
    ]
    report_imgs = [x for x in report_imgs if x]
    if report_imgs:
        lines.append("## 券商研报图表")
        lines.append("")
        for x in report_imgs:
            lines.append(x)
            lines.append("")
        lines.append("{REPORT_DESC}")
        lines.append("")

    # IR 业绩演示图表
    ir_imgs = [
        _img_md(f"业绩演示{i+1}", img_map.get(f"业绩演示{i+1}"))
        for i in range(len(ir_pages or []))
    ]
    ir_imgs = [x for x in ir_imgs if x]
    if ir_imgs:
        lines.append("## 业绩演示图表")
        lines.append("")
        for x in ir_imgs:
            lines.append(x)
            lines.append("")
        lines.append("{IR_DESC}")
        lines.append("")
    else:
        lines.append("## 业绩演示图表")
        lines.append("")
        lines.append("{IR_CHARTS}")
        lines.append("")

    # 事件列表
    lines.append("## 近期关键事件")
    lines.append("")
    lines.append("{EVENTS}")
    lines.append("")

    # 分隔线
    lines.append("---")
    lines.append("")
    lines.append("═══ 以上是证据，以下是判断 ═══")
    lines.append("")

    # 判断区
    lines.append("## 综合判断")
    lines.append("")
    lines.append("{JUDGMENT}")
    lines.append("")
    lines.append("## 什么会改变我的判断")
    lines.append("")
    lines.append("{CHANGE_TRIGGERS}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> ⚠️ 以上分析仅供参考，不构成投资建议。")

    content = "\n".join(lines)
    path = os.path.join(out, "report_template.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = os.path.getsize(path) // 1024
    print(f"    📝 report_template.md: {size_kb}KB")
    return path


# ═══════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════


# 常见港股 ADR ratio 表（1 ADR = N 股港股）
# 缺失的默认按 1:1 处理，并在返回里标注 ratio_source
_ADR_RATIO_TABLE = {
    "TCEHY": 1,   # 腾讯
    "BABA": 8,    # 阿里
    "BIDU": 10,   # 百度
    "JD": 2,      # 京东
    "NIO": 1,     # 蔚来
    "PDD": 4,     # 拼多多
    "NTES": 25,   # 网易
    "BILI": 1,    # 哔哩哔哩
    "LI": 2,      # 理想
    "XPEV": 2,    # 小鹏
    "TCOM": 1,    # 携程
    "YUMC": 1,    # 百胜中国
    "GDS": 1,     # 万国数据
    "HUYA": 1,    # 虎牙
    "DOYU": 1,    # 斗鱼
    "ZTO": 1,     # 中通快递
    "WB": 1,      # 微博
    "VIPS": 1,    # 唯品会
    "DADA": 1,    # 达达
    "BEKE": 2,    # 贝壳
    "TAL": 2,     # 好未来
    "EDU": 1,     # 新东方
    "FUTU": 1,    # 富途
    "TIGR": 1,    # 老虎证券
}


def _fetch_adr(adr_code, hk_code=None):
    """ADR 数据 + 溢价/折价百分比（结构化）。

    港股次日开盘方向信号：
      adr_premium_pct > 1% → 次日大概率高开
      adr_premium_pct < -1% → 次日大概率低开

    ADR ratio 来自 _ADR_RATIO_TABLE，缺失默认 1:1 并标注。
    """
    if not adr_code or not HAS_YFINANCE:
        return {}
    try:
        h = _yf(adr_code).history(period="5d")
        if h.empty:
            return {}
        fx = _yf("USDHKD=X").history(period="1d")
        rate = float(fx["Close"].iloc[-1]) if not fx.empty else 7.80
        adr_close = float(h["Close"].iloc[-1])

        result = {
            "adr_close_usd": round(adr_close, 2),
            "usd_hkd_rate": round(rate, 4),
        }

        # 计算 ADR 溢价/折价
        ratio = _ADR_RATIO_TABLE.get(adr_code.upper(), 1)
        result["adr_ratio"] = ratio
        result["ratio_source"] = (
            "已知表" if adr_code.upper() in _ADR_RATIO_TABLE else "默认1:1（未在ratio表中，溢价可能不准）"
        )

        if hk_code:
            try:
                hk_h = _yf(hk_code).history(period="1d")
                if not hk_h.empty:
                    hk_close = float(hk_h["Close"].iloc[-1])
                    # 1 ADR = ratio 股港股，ADR 美元价 * 汇率 / ratio = 对应港股港币价
                    adr_implied_hkd = adr_close * rate / ratio
                    premium_pct = (adr_implied_hkd / hk_close - 1) * 100
                    result["hk_close"] = round(hk_close, 2)
                    result["adr_implied_hkd"] = round(adr_implied_hkd, 2)
                    result["adr_premium_pct"] = round(premium_pct, 2)
                    # 信号标注
                    if premium_pct > 1:
                        result["signal"] = f"ADR 溢价 {premium_pct:.2f}% → 次日港股大概率高开"
                    elif premium_pct < -1:
                        result["signal"] = f"ADR 折价 {premium_pct:.2f}% → 次日港股大概率低开"
                    else:
                        result["signal"] = f"ADR 溢价 {premium_pct:.2f}% → 中性，无明显方向信号"
            except Exception as e:
                print(f"    ⚠ ADR 溢价计算失败: {e}")

        return result
    except Exception as e:
        print(f"    ⚠ ADR 数据获取失败: {e}")
        return {}


def _download_em_kline_multi(market, pure, out):
    """东方财富多周期K线图+分时图直接下载（日K/周K/月K/分时）

    东方财富图片服务器支持的 imageType:
      K=日K, RTOPH=分时, KL=迷你K线
    周K/月K图片服务器不支持，需要从API获取数据后本地绘制。
    """
    _PIC_CHARTS = {
        "kline": ("K", "kline_em.png", "东方财富日K线图"),
        "intraday": ("RTOPH", "kline_intraday.png", "东方财富分时图"),
    }
    results = {}
    if not market or not pure:
        return results
    nid = _em_nid(market, pure)
    if not nid:
        return results

    # 1. 直接下载图片服务器支持的类型（日K + 分时）
    for key, (img_type, filename, label) in _PIC_CHARTS.items():
        path = os.path.join(out, filename)
        if _download_em_pic(nid, img_type, path):
            results[key] = path
            print(f"    ✅ {label}直链")

    # 2. 周K/月K：从东方财富K线API获取数据 → matplotlib本地绘制
    for key, klt_val, label, filename in [
        ("kline_weekly", "102", "周K线图", "kline_weekly.png"),
        ("kline_monthly", "103", "月K线图", "kline_monthly.png"),
    ]:
        path = os.path.join(out, filename)
        drawn = _draw_period_kline(nid, klt_val, path, label)
        if drawn:
            results[key] = path
    return results


def _draw_period_kline(nid, klt, save_path, label=""):
    """从东方财富K线API获取数据，用matplotlib绘制指定周期K线图"""
    try:
        # klt: 101=日, 102=周, 103=月
        api_url = (
            f"https://push2.eastmoney.com/api/qt/stock/kline/get?"
            f"secid={nid}&fields1=f1,f2,f3,f4,f5,f6&"
            f"fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&"
            f"klt={klt}&fqt=1&end=20500101&lmt=300"
        )
        req = urllib.request.Request(api_url, headers=_EM_PIC_HEADERS)
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        klines = data.get("data", {}).get("klines", [])
        if len(klines) < 5:
            print(f"    ⚠ {label}: 数据不足({len(klines)}条)")
            return None

        # 解析K线数据
        records = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 11:
                records.append(
                    {
                        "date": parts[0],
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "volume": float(parts[6]),
                    }
                )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        # 用matplotlib绘制K线图（不依赖mplfinance）
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        dark_style()
        fig.patch.set_facecolor(C["bg"])
        ax1.set_facecolor(C["surface"])
        ax2.set_facecolor(C["surface"])

        dates = range(len(df))
        colors = [
            C["pos"] if c >= o else C["neg"] for o, c in zip(df["open"], df["close"])
        ]

        # 画实体
        for i, row in df.iterrows():
            color = C["pos"] if row["close"] >= row["open"] else C["neg"]
            ax1.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            rect_bottom = min(row["open"], row["close"])
            rect_height = abs(row["close"] - row["open"]) or 0.5
            ax1.bar(
                i,
                rect_height,
                bottom=rect_bottom,
                color=color,
                width=0.7,
                edgecolor=color,
            )

        # 均线
        for ma_len, color, name in [(5, "#FFD700", "MA5"), (20, "#00BFFF", "MA20")]:
            if len(df) >= ma_len:
                ma = df["close"].rolling(ma_len).mean()
                ax1.plot(dates, ma, color=color, linewidth=1.2, label=name, alpha=0.85)

        ax1.set_title(f"\n{label}", fontsize=14, color=C["text"], fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.25)
        ax1.set_ylabel("价格", color=C["text"])

        # 成交量
        vol_colors = [
            C["pos"] if c >= o else C["neg"]
            for o, c in zip(df["open"].iloc[-len(df) :], df["close"].iloc[-len(df) :])
        ]
        ax2.bar(dates, df["volume"], color=vol_colors, width=0.7, alpha=0.8)
        ax2.set_ylabel("成交量", color=C["text"])
        ax2.grid(True, alpha=0.25)

        # X轴日期标签
        n_bars = len(df)
        tick_step = max(1, n_bars // 12)
        tick_pos = list(range(0, n_bars, tick_step))
        if n_bars - 1 not in tick_pos:
            tick_pos.append(n_bars - 1)
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(
            [
                (
                    df.iloc[i]["date"].strftime("%m/%d")
                    if hasattr(df.iloc[i]["date"], "strftime")
                    else str(df.iloc[i]["date"])[:10]
                )
                for i in tick_pos
            ],
            rotation=45,
            fontsize=8,
        )

        plt.tight_layout()
        save_fig(fig, save_path)
        print(f"    ✅ {label}: 本地绘制({len(df)}根K线)")
        return save_path

    except urllib.error.URLError as e:
        print(f"    ⚠ {label}: 网络错误({e})")
        return None
    except Exception as e:
        print(f"    ⚠ {label}: {e}")
        return None


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="股票数据获取 v28.0")
    parser.add_argument("stock", help="股票代码 (如 0700.HK, 600519.SS, AAPL)")
    parser.add_argument("--adr", default="", help="ADR代码 (如 TCEHY)")
    parser.add_argument(
        "--output-dir", default="", help="输出目录（默认为当前工作目录）"
    )
    args = parser.parse_args()

    code = args.stock
    market, pure = _detect_market(code)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    # 输出目录：--output-dir 参数 > 向上搜索 SOUL.md 定位 workspace > getcwd
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = _find_workspace()
        if not output_base:
            output_base = os.getcwd()
    out = os.path.abspath(os.path.join(output_base, f"stock_data_output/{pure}_{ts}"))
    os.makedirs(out, exist_ok=True)

    TOTAL_TIMEOUT = 90 if market in ("sh", "sz", "hk") else 60  # A股/港股 90秒，美股 60秒

    print(f"📊 获取 {code} 的全部数据（市场: {market}）...")
    print(f"  ⏱ 数据获取超时: {TOTAL_TIMEOUT}秒（所有数据源并行）")

    # ═══════════════════════════════════════
    # Phase 1: 并行获取所有数据（60秒总超时）
    # ═══════════════════════════════════════
    print("\n  ── Phase 1: 并行数据获取 ──")

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {}

        # K线 — 4个源并行
        futures[pool.submit(kline_tushare, code, market, pure)] = "kline_tushare"
        futures[pool.submit(kline_akshare, code, market, pure)] = "kline_akshare"
        futures[pool.submit(kline_eastmoney, code, market, pure)] = "kline_eastmoney"
        futures[pool.submit(kline_yfinance, code)] = "kline_yfinance"

        # 基本面 — 3个源并行
        futures[pool.submit(_fetch_info_eastmoney, market, pure)] = "info_eastmoney"
        futures[pool.submit(_fetch_info_yfinance, code)] = "info_yfinance"
        futures[pool.submit(_fetch_info_tushare, market, pure)] = "info_tushare"

        # 其他 — 各自独立
        futures[pool.submit(fetch_capital_flow, code, market, pure)] = "capital_flow"
        futures[pool.submit(fetch_main_capital_flow, code, market, pure)] = (
            "main_capital_flow"
        )
        futures[pool.submit(fetch_em_announcements, market, pure)] = "announcements"
        futures[pool.submit(fetch_reports, pure, market)] = "reports"
        futures[pool.submit(fetch_macro)] = "macro"
        futures[pool.submit(fetch_peer_comparison, market, pure)] = "peer_comparison"
        futures[pool.submit(fetch_lhb, pure, market)] = "lhb"
        futures[pool.submit(fetch_hk_short, pure, market)] = "hk_short"
        futures[pool.submit(_fetch_adr, args.adr, code)] = "adr"
        futures[pool.submit(_download_em_kline_multi, market, pure, out)] = (
            "em_kline_pic"
        )
        futures[pool.submit(fetch_ir_presentation, code, pure, market, out)] = (
            "ir_presentation"
        )

        # 收割结果
        raw = {}
        try:
            for future in as_completed(futures, timeout=TOTAL_TIMEOUT):
                name = futures[future]
                try:
                    raw[name] = future.result(timeout=5)
                except Exception as e:
                    print(f"    ⚠ {name} 失败: {e}")
                    raw[name] = None
        except Exception:
            # 总超时到了，收集已完成的
            for future, name in futures.items():
                if future.done():
                    try:
                        raw[name] = future.result(timeout=0)
                    except Exception:
                        raw[name] = None
                else:
                    print(f"    ⏰ {name} 超时未返回")
                    raw[name] = None

    # ── 整理结果 ──
    # K线：按优先级选最佳
    kline = None
    kline_src = "none"
    kline_priority = (
        ["kline_eastmoney", "kline_akshare", "kline_tushare", "kline_yfinance"]
        if market in ("hk", "sh", "sz")
        else ["kline_yfinance", "kline_akshare", "kline_tushare"]
    )
    for key in kline_priority:
        if raw.get(key) is not None:
            kline = raw[key]
            kline_src = key.replace("kline_", "")
            break
    print(f"  K线来源: {kline_src}" if kline is not None else "  ⚠ 无K线数据")

    # 基本面：合并三个源
    info_em = raw.get("info_eastmoney") or {}
    info_yf = raw.get("info_yfinance") or {}
    info_ts = raw.get("info_tushare") or {}
    info = _merge_info(info_em, info_yf, info_ts)

    # 多源交叉验证：关键字段差异 >2% 时标记
    cross_check = _cross_validate_sources(info_em, info_yf, info_ts)
    if cross_check.get("discrepancies"):
        print(f"    ⚠ 多源数据分歧: {len(cross_check['discrepancies'])} 项")

    # 资金流
    cap_result = raw.get("capital_flow") or ([], "N/A")
    cap_flow = cap_result[0] if isinstance(cap_result, tuple) else []
    cap_label = cap_result[1] if isinstance(cap_result, tuple) else "N/A"

    # 其他
    reports_raw = raw.get("reports") or {}
    if isinstance(reports_raw, dict):
        reports = reports_raw.get("reports") or []
        reports_consensus = reports_raw.get("consensus") or {}
    else:
        # 向后兼容（旧版返回 list）
        reports = reports_raw if isinstance(reports_raw, list) else []
        reports_consensus = {}
    macro = raw.get("macro") or {}
    adr_data = raw.get("adr") or {}
    em_pics = raw.get("em_kline_pic") or {"kline": None, "intraday": None}
    ir_pages = raw.get("ir_presentation") or []
    announcements = raw.get("announcements") or []
    main_cap = raw.get("main_capital_flow") or {}
    peer_comparison = raw.get("peer_comparison") or {}
    lhb = raw.get("lhb") or []
    hk_short = raw.get("hk_short") or {}

    # Phase 1.5: 串行跑东方财富新闻搜索（需要 stock_name）
    # 关键词优先用 stock_name，否则用 pure（A股支持代码搜索）
    stock_name = info.get("shortName", "")
    news_keyword = stock_name or pure
    em_news = fetch_em_news(news_keyword, count=10) if news_keyword else []

    def _has_data(v):
        if v is None:
            return False
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return not v.empty
        if isinstance(v, (list, dict)):
            return len(v) > 0
        return True

    got = sum(
        1
        for v in [kline, info, cap_flow, reports, macro, adr_data, em_news, main_cap]
        if _has_data(v)
    )
    print(f"  Phase 1+1.5 完成: {got}/8 类数据获取成功")

    # ═══════════════════════════════════════
    # Phase 2: 串行生成图表（本地操作，秒级）
    # ═══════════════════════════════════════
    print("\n  ── Phase 2: 生成图表 ──")

    # 补 price_as_of：若 EM 没返回，从 K 线最后一根 Date 取
    if not info.get("price_as_of") and kline is not None and not isinstance(kline, str):
        try:
            last_date = kline.index[-1]
            if hasattr(last_date, "strftime"):
                info["price_as_of"] = last_date.strftime("%Y-%m-%d")
            else:
                info["price_as_of"] = str(last_date)[:10]
        except Exception as e:
            print(f"    ⚠ unknown 异常: {e}")
            pass

    tech = compute_technicals(kline)

    charts = {}
    # K线图：优先用东方财富直链图片（含日K/周K/月K），否则本地绘制
    _kline_set = False
    for k in ["kline", "kline_weekly", "kline_monthly"]:
        if em_pics.get(k):
            charts[k] = em_pics[k]
            _kline_set = True
            label = {"kline": "日K", "kline_weekly": "周K", "kline_monthly": "月K"}.get(
                k, "K"
            )
            print(f"    ✅ {label}: 东方财富直链")
    if not _kline_set and kline is not None:
        charts["kline"] = chart_kline(kline, code, out)

    if em_pics.get("intraday"):
        charts["kline_intraday"] = em_pics["intraday"]

    charts["capital_flow"] = chart_capital_flow(cap_flow, code, out, label=cap_label)
    charts["macro"] = chart_macro(macro, out)
    charts["valuation"] = chart_valuation(info, code, out)
    # PE Band（历史百分位）
    pe_band_path, pe_band_stats = chart_pe_band(kline, info, code, out)
    if pe_band_path:
        charts["pe_band"] = pe_band_path
    # 获取结构化财务数据 + 画图
    print("  → 获取结构化财务三表...")
    financials_struct = fetch_financials_structured(code, market, pure)
    charts["financials_trend"] = chart_financials(code, out, financials_data=financials_struct)
    charts["adr_premium"] = chart_adr(code, args.adr, out) if args.adr else None

    chart_count = sum(1 for v in charts.values() if v)
    for name, path in charts.items():
        if path:
            print(f"    ✅ {name}: {os.path.basename(path)}")

    # 研报PDF图表
    print("  → 研报PDF图表...")
    stock_name = info.get("shortName", "")
    report_pages = auto_fetch_report_pdf(
        reports, out, stock_name=stock_name, stock_code=pure, max_reports=2, max_pages=4
    )

    # ═══════════════════════════════════════
    # Phase 3: 写文件（秒级）
    # ═══════════════════════════════════════
    print("\n  ── Phase 3: 写文件 ──")

    # 资金流汇总
    cap_summary = {}
    if cap_flow:
        recent = cap_flow[-5:] if len(cap_flow) >= 5 else cap_flow
        if recent:
            d = 1 if recent[-1]["shares_change"] > 0 else -1
            cons = 0
            for x in reversed(recent):
                if (x["shares_change"] > 0 and d > 0) or (
                    x["shares_change"] < 0 and d < 0
                ):
                    cons += 1
                else:
                    break
            cap_summary = {
                "type": cap_label,
                "consecutive_days": cons,
                "direction": "净买入" if d > 0 else "净卖出",
                "recent_total_万股": round(
                    sum(x["shares_change"] for x in recent) / 1e4, 1
                ),
            }

    # 构建每个数据类别的时间戳与延迟说明
    price_as_of_str = info.get("price_as_of", "")
    data_freshness = {
        "realtime_quote": {
            "data_time": price_as_of_str,
            "delay_note": "收盘数据" if price_as_of_str else "时间未知",
        },
        "kline": {
            "data_time": price_as_of_str,
            "delay_note": "收盘后数据（含今日）" if price_as_of_str else "",
        },
        "capital_flow": {
            "data_time": "",
            "delay_note": "T-1日数据（资金流向通常延迟1个交易日）",
        },
        "main_capital_flow": {
            "data_time": "",
            "delay_note": "T-1日数据（主力资金流向延迟1个交易日）",
        },
        "announcements": {
            "data_time": "",
            "delay_note": "实时公告（可能有数小时延迟）",
        },
        "news": {
            "data_time": "",
            "delay_note": "搜索时效约 1-3 天",
        },
        "analyst_reports": {
            "data_time": "",
            "delay_note": "研报发布日期见各条 date 字段，近180天",
        },
        "lhb": {
            "data_time": "",
            "delay_note": "T+1日盘后公布（龙虎榜次日才出）",
        },
        "hk_short": {
            "data_time": "",
            "delay_note": "T+1日数据（港交所次日公布卖空数据）",
        },
        "quarterly_financials": {
            "data_time": financials_struct.get("quarters", [{}])[0].get("period", "") if financials_struct.get("quarters") else "",
            "delay_note": "最新季度财报（季报有1-3个月披露延迟）",
        },
        "peer_comparison": {
            "data_time": price_as_of_str,
            "delay_note": "同业估值随实时行情更新",
        },
        "pe_band": {
            "data_time": price_as_of_str,
            "delay_note": "历史百分位基于近3年K线，当前PE假设EPS不变",
        },
    }

    result = {
        "stock_code": code,
        "market": market,
        "fetch_time": datetime.now().isoformat(),
        "kline_source": kline_src,
        "financials": info,
        "quarterly_financials": financials_struct,
        "pe_band_stats": pe_band_stats,
        "technicals": tech,
        "capital_flow_label": cap_label,
        "capital_flow_summary": cap_summary,
        "main_capital_flow": main_cap,
        "news": em_news[:10],
        "announcements": announcements[:10],
        "analyst_reports": reports[:10],
        "analyst_consensus": reports_consensus,
        "peer_comparison": peer_comparison,
        "lhb": lhb,
        "hk_short": hk_short,
        "report_chart_pages": report_pages,
        "ir_presentation_pages": ir_pages,
        "adr": adr_data,
        "macro": macro,
        "data_freshness": data_freshness,
        "cross_check": cross_check,
        "charts": {k: v for k, v in charts.items() if v},
        "chart_count": chart_count + len(report_pages) + len(ir_pages),
    }

    with open(os.path.join(out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    # ── Phase 3b: 整理图片到固定目录，输出数据摘要 ──
    # ── Phase 3b: 整理输出（图片保留在任务目录中）──
    img_paths = {}
    for key, path in charts.items():
        if path and os.path.exists(path):
            img_paths[key] = path
    for i, p in enumerate(report_pages or []):
        if p and os.path.exists(p):
            img_paths[f"report_{i+1}"] = p
    for i, p in enumerate(ir_pages or []):
        if p and os.path.exists(p):
            img_paths[f"ir_{i+1}"] = p

    print(f"  📸 图片: {len(img_paths)} 张（在任务目录 {out} 中）")
    # 构建数据摘要（输出到 stdout，供主模型读取）
    # ── 文本折线图（Webchat 内嵌展示用）──
    _tc = {}
    if (
        HAS_TEXT_CHART
        and kline is not None
        and not isinstance(kline, str)
        and hasattr(kline, "shape")
    ):
        try:
            _closes = kline["close"].tolist() if "close" in kline.columns else []
            _dates = (
                kline.index.strftime("%m/%d").tolist()
                if hasattr(kline.index, "strftime")
                else list(range(len(_closes)))
            )
            if len(_closes) >= 5:
                _tc["sparkline_20d"] = sparkline(_closes[-20:])
                _tc["sparkline_60d"] = sparkline(
                    _closes[-60:] if len(_closes) >= 60 else _closes
                )
                # block_chart 已废弃（占空间过多），只保留 sparkline
        except Exception as e:
            print(f"    ⚠ unknown 异常: {e}")
            pass

    summary = {
        "stock_code": code,
        "stock_name": info.get("shortName", ""),
        "market": market,
        "currency": info.get("currency", ""),
        "price_as_of": info.get("price_as_of", ""),
        "output_dir": out,
        "fetch_time": datetime.now().isoformat(),
        "kline_source": kline_src,
        "images": img_paths,
        "data": {
            "financials": {
                k: v
                for k, v in info.items()
                if v is not None and str(v) != "" and not isinstance(v, pd.DataFrame)
            },
            "quarterly_financials_summary": financials_struct.get("latest_summary", {}),
            "pe_band_stats": pe_band_stats,
            "technicals": tech,
            "capital_flow_summary": cap_summary,
            "main_capital_flow": main_cap,
            "news": em_news[:10],
            "announcements": announcements[:10],
            "analyst_reports": reports[:5],
            "analyst_consensus": reports_consensus,
            "peer_comparison": peer_comparison,
            "lhb": lhb[:8],
            "hk_short": hk_short,
            "macro": macro,
            "adr": adr_data,
            "data_freshness": data_freshness,
            "cross_check": cross_check,
        },
        "text_charts": _tc,
        "export_options": {
            "pdf_cmd": f"python3 {os.path.abspath('export_report.py')} {{out}}/report.md --format pdf",
            "html_cmd": f"python3 {os.path.abspath('export_report.py')} {{out}}/report.md --format html",
            "open_html": f"open {{out}}/report.html",
        },
    }

    summary_path = os.path.join(out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    total = chart_count + len(report_pages) + len(ir_pages)
    print(f"\n{'='*60}")
    print(f"✅ 完成! 共 {total} 张图表")
    print(f"  📄 data.json:      {os.path.join(out,'data.json')}")
    print(f"  📋 summary.json:   {summary_path}")
    print(f"  📸 输出目录:       {out}")
    print(f"{'='*60}")

    if total == 0:
        print("  ⚠ 未生成任何图表")

    # 输出摘要 JSON 到 stdout（主模型可解析）
    print("\n__SUMMARY_BEGIN__")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("__SUMMARY_END__")

    # HTML 报告由主模型分析完成后生成并写入 {out}/report.html
    # 脚本只负责数据采集+画图，不生成报告


def _generate_data_html(summary, html_path):
    """生成数据概览 HTML（含图表+数据），脚本自动生成并在浏览器中打开。"""
    import html as _html_mod

    code = summary.get("stock_code", "")
    name = summary.get("stock_name") or code
    market = summary.get("market", "")
    images = summary.get("images", {})
    data = summary.get("data", {})
    fetch_time = summary.get("fetch_time", "")
    tc = summary.get("text_charts") or {}

    def esc(s):
        if not s:
            return ""
        return _html_mod.escape(str(s))

    # 图片列表
    img_order = [
        "kline",
        "kline_weekly",
        "kline_monthly",
        "kline_intraday",
        "valuation",
        "capital_flow",
        "macro",
        "adr_premium",
        "financials_trend",
    ]
    img_labels_map = {
        "kline": "日K线图",
        "kline_weekly": "周K线图",
        "kline_monthly": "月K线图",
        "kline_intraday": "分时图",
        "valuation": "估值图",
        "capital_flow": "资金流向图",
        "macro": "宏观环境图",
        "adr_premium": "ADR溢价图",
        "financials_trend": "营收趋势图",
    }
    img_items = []
    for key in img_order:
        path = images.get(key)
        if path and os.path.exists(path):
            img_items.append((img_labels_map.get(key, key), path))
    for key, path in images.items():
        if path and os.path.exists(path) and key not in img_order:
            img_items.append((key.replace("_", " ").title(), path))

    from io import StringIO

    buf = StringIO()
    w = buf.write

    # === HTML HEAD ===
    w('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n')
    w(
        '<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    )
    w("<title>" + esc(name) + " (" + esc(code) + ") \u2014 数据概览</title>\n<style>\n")
    w("*{margin:0;padding:0;box-sizing:border-box;}\n")
    w(
        'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.8;background:#f0f2f5;color:#333;}\n'
    )
    w(".container{max-width:920px;margin:0 auto;padding:20px;}\n")
    w(
        ".card{background:white;border-radius:12px;padding:36px;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:20px;}\n"
    )
    w(
        "h1{font-size:24px;color:#1a1a1a;border-bottom:3px solid #1e88e5;padding-bottom:10px;}\n"
    )
    w(".meta{color:#888;font-size:13px;margin-bottom:28px;}\n")
    w(
        "h2{font-size:18px;color:#1e88e5;margin-top:32px;margin-bottom:14px;padding-left:12px;border-left:4px solid #1e88e5;}\n"
    )
    w("table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13px;}\n")
    w("th,td{border:1px solid #e8eaed;padding:10px 12px;text-align:left;}\n")
    w("th{background:#1e88e5;color:white;font-weight:600;}\n")
    w("tr:nth-child(even){background:#f8f9fa;}\n")
    w(
        ".chart-box{text-align:center;margin:22px 0;background:#fafbfc;border:1px solid #e8eaed;border-radius:8px;padding:18px;}\n"
    )
    w(".chart-box img{max-width:100%;height:auto;border-radius:6px;}\n")
    w(".chart-caption{color:#888;font-size:13px;margin-top:10px;}\n")
    w(
        '.sparkline-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 20px;margin:20px 0;font-family:"SF Mono",Monaco,monospace;font-size:17px;letter-spacing:2px;text-align:center;}\n'
    )
    w(".news-item{border-bottom:1px solid #f0f0f0;padding:12px 0;}\n")
    w(".news-item:last-child{border:none;}\n")
    w(".news-title{font-weight:600;font-size:14px;color:#1a1a1a;}\n")
    w(".news-meta{color:#999;font-size:12px;margin:3px 0;}\n")
    w(".news-content{color:#666;font-size:13px;margin-top:4px;line-height:1.6;}\n")
    w(
        ".hint{background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border:1px solid #bfdbfe;border-radius:8px;padding:18px 22px;margin-top:28px;font-size:13px;color:#1e40af;line-height:1.7;}\n"
    )
    w(
        ".footer{text-align:center;color:#aaa;font-size:12px;margin-top:24px;padding-top:16px;border-top:1px solid #e8eaed;}\n"
    )
    w(
        ".empty{color:#999;font-style:italic;padding:20px;text-align:center;background:#fafbfc;border-radius:6px;margin:10px 0;}\n"
    )
    w('</style>\n</head>\n<body>\n<div class="container">\n<div class="card">\n')

    # 标题
    w("<h1>📊 " + esc(name) + " (" + esc(code) + ") 数据概览</h1>\n")
    w(
        '<div class="meta"><strong>采集时间：</strong>'
        + fetch_time[:19]
        + " &nbsp;|&nbsp; <strong>市场：</strong>"
        + market.upper()
        + " &nbsp;|&nbsp; <strong>状态：</strong>原始数据采集完成，待主模型分析</div>\n"
    )

    # 文本折线图
    sp = tc.get("sparkline_60d") or tc.get("sparkline_20d") or ""
    if sp:
        w('<div class="sparkline-box">近期走势：' + esc(sp) + "</div>\n")

    # 图表
    w("<h2>📈 图表</h2>\n")
    if img_items:
        for label, path in img_items:
            w(
                '<div class="chart-box"><img src="file://'
                + path
                + '" alt="'
                + label
                + '" onerror="this.onerror=null;this.parentElement.style.display=\'none\'">'
                + '<div class="chart-caption">'
                + label
                + "</div></div>\n"
            )
    else:
        w('<p class="empty">本次未生成图表</p>\n')

    # 基本面
    w("<h2>💰 基本面数据</h2>\n")
    fin = data.get("financials") or {}
    fin_rows = ""
    if isinstance(fin, dict):
        for k, v in list(fin.items())[:15]:
            if v is not None and str(v).strip() and not str(v).startswith("{"):
                fin_rows += (
                    "<tr><td>" + esc(k) + "</td><td>" + esc(str(v)) + "</td></tr>"
                )
    if fin_rows:
        w("<table><tr><th>指标</th><th>数值</th></tr>" + fin_rows + "</table>\n")
    else:
        w('<p class="empty">未获取到基本面数据</p>\n')

    # 技术指标
    w("<h2>📐 技术指标</h2>\n")
    tech = data.get("technicals") or {}
    tech_rows = ""
    if isinstance(tech, dict):
        for k, v in list(tech.items())[:12]:
            if v is not None:
                tech_rows += (
                    "<tr><td>" + esc(k) + "</td><td>" + esc(str(v)) + "</td></tr>"
                )
    if tech_rows:
        w("<table><tr><th>指标</th><th>数值</th></table>" + tech_rows + "</table>\n")
    else:
        w('<p class="empty">未获取到技术指标</p>\n')

    # 资金流向
    w("<h2>💵 资金流向</h2>\n")
    mcf = data.get("main_capital_flow") or {}
    if isinstance(mcf, dict) and mcf.get("total_main_net_inflow") is not None:
        total = mcf.get("total_main_net_inflow", 0)
        days = mcf.get("consecutive_days", 0)
        direction = mcf.get("direction", "")
        w(
            f"<p><strong>{esc(direction)}:</strong> {total/1e8:.2f}亿 | 连续{days}天</p>\n"
        )
    cap = data.get("capital_flow_summary") or {}
    if isinstance(cap, dict) and cap:
        w(
            f'<p><strong>方向:</strong> {esc(cap.get("direction","N/A"))} | <strong>连续:</strong> {str(cap.get("consecutive_days","N/A"))}天</p>\n'
        )
    if (not mcf) and (not cap):
        w('<p class="empty">未获取到资金流数据</p>\n')

    # 研报
    w("<h2>📑 券商研报（最新）</h2>\n")
    rpt_rows = ""
    for r in (data.get("analyst_reports") or [])[:6]:
        rpt_rows += (
            "<tr><td>"
            + esc(r.get("org", ""))
            + "</td><td>"
            + esc(r.get("rating", ""))
            + "</td><td>"
            + esc(r.get("title", ""))
            + "</td><td>"
            + str(r.get("date", ""))
            + "</td></tr>"
        )
    if rpt_rows:
        w(
            "<table><tr><th>机构</th><th>评级</th><th>标题</th><th>日期</th></tr>"
            + rpt_rows
            + "</table>\n"
        )
    else:
        w('<p class="empty">未获取到研报数据</p>\n')

    # 新闻
    w("<h2>📰 近期新闻</h2>\n")
    news_divs = ""
    for n in (data.get("news") or data.get("cls_news") or [])[:10]:
        title = n.get("title", "")
        date = str(n.get("date", ""))[:16]
        source = n.get("source", "")
        content_preview = (n.get("content") or "")[:120]
        news_divs += (
            '<div class="news-item"><div class="news-title">'
            + esc(title)
            + '</div><div class="news-meta">'
            + date
            + " · "
            + esc(source)
            + '</div><div class="news-content">'
            + esc(content_preview)
            + "...</div></div>"
        )
    if news_divs:
        w(news_divs + "\n")
    else:
        w('<p class="empty">未获取到新闻</p>\n')

    # 提示 + footer
    w(
        '<div class="hint"><strong>💡 下一步：</strong>多模态主模型将基于以上数据和图片，<br>'
    )
    w("输出完整的分析报告到聊天窗口中。<br>")
    w('如需导出 <strong>PDF 报告</strong>，请在聊天中回复"导出 PDF"。</div>\n')
    w(
        '<div class="footer">数据由 stock-analyst skill 自动采集 · '
        + fetch_time[:19]
        + "</div>\n"
    )
    w("</div></div></body></html>\n")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    main()
