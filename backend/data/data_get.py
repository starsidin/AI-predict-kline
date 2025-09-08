# ==============================
# File: binance_downloader_cfg.py
# ==============================
# 用法：
#   1) 将上面的 config.example.yaml 复制为 config.yaml 并修改参数
#   2) 运行：python binance_downloader_cfg.py --config config.yaml
#
# 功能：
#   - 按 config 批量下载 Binance Vision 月度数据（spot / futures-um / futures-cm / options）
#   - 支持 klines / trades / aggTrades
#   - 支持并发下载、跳过已存在、可选 CHECKSUM 校验
#   - 可选自动解压 zip，并按月合并为单一 CSV（适合AI训练）
#
# 目录结构（示例）：
#   {out}/{market}/{datatype}/{symbol}/[interval]/YYYY-MM.zip
#   解压后默认在同目录下生成对应csv；merge=true则生成 merged_{symbol}[_interval].csv

import argparse
import concurrent.futures as cf
from dataclasses import dataclass
from datetime import datetime
import os
import sys
import time
import hashlib
import requests
import zipfile
import io
import glob
from typing import List, Optional, Tuple, Dict, Any

try:
    import yaml  # pip install pyyaml
except Exception as e:
    yaml = None
    print("[WARN] 未安装 pyyaml，将无法读取YAML配置。请先：pip install pyyaml", file=sys.stderr)

BASE = "https://data.binance.vision"
VALID_INTERVALS = {"1s","1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}
MARKET_MAP = {
    "spot": ("spot", "Spot"),
    "futures-um": ("futures/um", "USDT-M Futures"),
    "futures-cm": ("futures/cm", "COIN-M Futures"),
    "options": ("option", "Options"),
}
DATATYPE_DIR = {
    "klines": "klines",
    "trades": "trades",
    "aggTrades": "aggTrades",
}

@dataclass
class Task:
    url: str
    url_checksum: Optional[str]
    save_path: str
    save_checksum_path: Optional[str]
    year: int
    month: int
    market: str
    datatype: str
    symbol: str
    interval: Optional[str]
    unzip: bool = False


def month_range(start: str, end: str) -> List[Tuple[int, int]]:
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    if s > e:
        raise ValueError("start 不能大于 end")
    res = []
    y, m = s.year, s.month
    while (y < e.year) or (y == e.year and m <= e.month):
        res.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return res


def monthly_filename(symbol: str, datatype: str, year: int, month: int, interval: Optional[str]) -> str:
    mm = f"{month:02d}"
    if datatype == "klines":
        if not interval:
            raise ValueError("klines 必须提供 interval")
        return f"{symbol}-{interval}-{year}-{mm}.zip"
    elif datatype == "trades":
        return f"{symbol}-trades-{year}-{mm}.zip"
    elif datatype == "aggTrades":
        return f"{symbol}-aggTrades-{year}-{mm}.zip"
    else:
        raise ValueError(f"不支持的数据类型: {datatype}")


def build_monthly_url(market_key: str, datatype: str, symbol: str, year: int, month: int, interval: Optional[str]) -> Tuple[str, str, str]:
    if market_key not in MARKET_MAP:
        raise ValueError(f"未知市场: {market_key}")
    market_path, _ = MARKET_MAP[market_key]
    if datatype not in DATATYPE_DIR:
        raise ValueError(f"未知数据类型: {datatype}")
    dtype_dir = DATATYPE_DIR[datatype]

    fname = monthly_filename(symbol, datatype, year, month, interval)
    csum = fname + ".CHECKSUM"

    if datatype == "klines":
        if not interval:
            raise ValueError("klines 必须提供 interval")
        prefix = f"/data/{market_path}/monthly/{dtype_dir}/{symbol}/{interval}"
        subdir = os.path.join(market_key, datatype, symbol, interval)
    else:
        prefix = f"/data/{market_path}/monthly/{dtype_dir}/{symbol}"
        subdir = os.path.join(market_key, datatype, symbol)

    data_url = f"{BASE}{prefix}/{fname}"
    checksum_url = f"{BASE}{prefix}/{csum}"
    return data_url, checksum_url, subdir


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def http_exists(url: str, timeout=10) -> bool:
    try:
        r = requests.head(url, timeout=timeout)
        if r.status_code == 200:
            return True
        r = requests.get(url, stream=True, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def download_file(url: str, save_path: str, retries: int = 3, backoff: float = 2.0, timeout: int = 30) -> bool:
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return True
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    raise requests.RequestException(f"HTTP {r.status_code}")
                total = int(r.headers.get("Content-Length", 0))
                tmp_path = save_path + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                if total and os.path.getsize(tmp_path) != total:
                    raise IOError("文件大小与Content-Length不一致")
                os.replace(tmp_path, save_path)
                return True
        except Exception as e:
            if i < retries - 1:
                time.sleep(backoff * (2 ** i))
            else:
                print(f"[ERROR] 下载失败: {url} -> {e}", file=sys.stderr)
    return False


def parse_checksum(content: str) -> Optional[str]:
    line = content.strip().splitlines()[0] if content.strip() else ""
    token = line.split()[0] if line else ""
    if all(c in "0123456789abcdefABCDEF" for c in token) and len(token) in (32, 40, 64):
        return token.lower()
    return None


def file_hash(path: str, algo: str = "md5", chunk=1024*1024) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_algo_by_len(hexhash: str) -> Optional[str]:
    if len(hexhash) == 32:
        return "md5"
    elif len(hexhash) == 40:
        return "sha1"
    elif len(hexhash) == 64:
        return "sha256"
    return None


def verify_checksum(zip_path: str, checksum_text: str) -> bool:
    expect = parse_checksum(checksum_text)
    if not expect:
        print(f"[WARN] 未能从 CHECKSUM 文本中解析出哈希，跳过校验。")
        return True
    algo = detect_algo_by_len(expect)
    if not algo:
        print(f"[WARN] 无法识别哈希算法，跳过校验。")
        return True
    actual = file_hash(zip_path, algo=algo)
    ok = (actual.lower() == expect.lower())
    if not ok:
        print(f"[ERROR] 校验失败: {os.path.basename(zip_path)} 期望 {expect} 实际 {actual}")
    return ok


def unzip_to_dir(zip_path: str, target_dir: str) -> List[str]:
    """解压 zip 到 target_dir，返回生成的文件列表（完整路径）。"""
    out_files: List[str] = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                # 避免ZipSlip - 修复Windows路径检查
                member_path = os.path.normpath(os.path.join(target_dir, member))
                target_abs = os.path.abspath(target_dir)
                member_abs = os.path.abspath(member_path)
                
                # 使用os.path.commonpath进行更可靠的路径检查
                try:
                    common = os.path.commonpath([target_abs, member_abs])
                    if not os.path.samefile(common, target_abs):
                        print(f"[WARN] 跳过可疑路径: {member}")
                        continue
                except (ValueError, OSError):
                    print(f"[WARN] 跳过无效路径: {member}")
                    continue
                    
                zf.extract(member, target_dir)
                out_files.append(member_path)
    except zipfile.BadZipFile:
        print(f"[ERROR] 压缩包损坏：{zip_path}")
    return out_files


def merge_csvs(csv_paths: List[str], merged_path: str):
    """将多个CSV按文件名自然顺序合并为一个（简单拼接，不去重）。"""
    if not csv_paths:
        return
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, 'w', encoding='utf-8') as wf:
        header_written = False
        for p in sorted(csv_paths):
            try:
                with open(p, 'r', encoding='utf-8') as rf:
                    for i, line in enumerate(rf):
                        if i == 0:
                            if not header_written:
                                wf.write(line)
                                header_written = True
                        else:
                            wf.write(line)
            except FileNotFoundError:
                continue
    print(f"[MERGED] -> {merged_path}")


def assemble_tasks(job: Dict[str, Any]) -> List[Task]:
    market = job['market']
    datatype = job['datatype']
    symbols = job['symbols'] if isinstance(job['symbols'], list) else [job['symbols']]
    start = job['start']
    end = job['end']
    interval = job.get('interval')
    out_dir = job.get('out', './binance_data')
    verify = bool(job.get('verify', False))
    unzip = bool(job.get('unzip', False))

    tasks: List[Task] = []
    for symbol in symbols:
        for (y, m) in month_range(start, end):
            data_url, checksum_url, subdir = build_monthly_url(market, datatype, symbol, y, m, interval)
            save_dir = os.path.join(out_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            fname = monthly_filename(symbol, datatype, y, m, interval)
            save_path = os.path.join(save_dir, fname)
            save_checksum_path = os.path.join(save_dir, fname + '.CHECKSUM') if verify else None
            tasks.append(Task(
                url=data_url,
                url_checksum=checksum_url if verify else None,
                save_path=save_path,
                save_checksum_path=save_checksum_path,
                year=y,
                month=m,
                market=market,
                datatype=datatype,
                symbol=symbol,
                interval=interval,
                unzip=unzip,
            ))
    return tasks


def worker(task: Task, verify: bool, timeout: int = 30) -> Tuple[str, bool]:
    if not http_exists(task.url, timeout=timeout):
        return (os.path.basename(task.save_path), False)

    ok = download_file(task.url, task.save_path, timeout=timeout)
    if not ok:
        return (os.path.basename(task.save_path), False)

    if verify and task.url_checksum and task.save_checksum_path:
        if http_exists(task.url_checksum, timeout=timeout):
            if download_file(task.url_checksum, task.save_checksum_path, timeout=timeout):
                try:
                    with open(task.save_checksum_path, 'r', encoding='utf-8', errors='ignore') as f:
                        checksum_text = f.read()
                    if not verify_checksum(task.save_path, checksum_text):
                        return (os.path.basename(task.save_path), False)
                except Exception as e:
                    print(f"[WARN] 读取/校验 CHECKSUM 出错: {task.save_checksum_path} -> {e}")
        else:
            print(f"[WARN] 未提供 CHECKSUM：{task.url_checksum}")

    # 可选自动解压
    if task.unzip:
        target_dir = os.path.dirname(task.save_path)
        _ = unzip_to_dir(task.save_path, target_dir)

    return (os.path.basename(task.save_path), True)


def run_job(job: Dict[str, Any]):
    name = job.get('name', 'job')
    market = job['market']
    datatype = job['datatype']
    workers = int(job.get('workers', 6))
    verify = bool(job.get('verify', False))
    timeout = int(job.get('timeout', 30))

    tasks = assemble_tasks(job)
    total = len(tasks)
    if total == 0:
        print(f"[SKIP] {name}: 没有可下载的月份范围。")
        return

    print(f"\n=== 开始任务: {name} ===\n"
          f"market={market}, datatype={datatype}, symbols={job.get('symbols')}, months={total}, "
          f"workers={workers}, verify={verify}, unzip={bool(job.get('unzip', False))}, merge={bool(job.get('merge', False))}")

    ok_cnt = 0
    fail_cnt = 0

    with cf.ThreadPoolExecutor(max_workers=workers) as exe:
        futs = [exe.submit(worker, t, verify, timeout) for t in tasks]
        for f in cf.as_completed(futs):
            name_, ok = f.result()
            if ok:
                ok_cnt += 1
                print(f"[OK] {name_}")
            else:
                fail_cnt += 1
                print(f"[FAIL] {name_}")

    print(f"任务 {name} 完成：成功 {ok_cnt}，失败 {fail_cnt}，总计 {total}")

    # 合并：针对每个 symbol（和可选 interval）合并当月CSV
    if job.get('merge'):
        out_dir = job.get('out', './binance_data')
        symbols = job['symbols'] if isinstance(job['symbols'], list) else [job['symbols']]
        interval = job.get('interval')
        for sym in symbols:
            # 目标目录与下载保持一致
            if datatype == 'klines' and interval:
                dl_dir = os.path.join(out_dir, market, datatype, sym, interval)
                merged_name = f"merged_{sym}_{interval}.csv"
            else:
                dl_dir = os.path.join(out_dir, market, datatype, sym)
                merged_name = f"merged_{sym}.csv"
            # 查找所有CSV（解压后通常在同一目录）
            csv_list = glob.glob(os.path.join(dl_dir, "*.csv"))
            if not csv_list:
                print(f"[MERGE] 未发现可合并CSV：{dl_dir}")
                continue
            merged_path = os.path.join(dl_dir, merged_name)
            merge_csvs(csv_list, merged_path)


def load_config(path: str) -> Dict[str, Any]:
    if not yaml:
        raise RuntimeError("未安装 pyyaml，无法读取配置文件。请先 pip install pyyaml")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or 'jobs' not in cfg:
        raise ValueError("配置文件格式错误：顶层应包含 jobs 列表")
    return cfg


def validate_job(job: Dict[str, Any]):
    required = ['market', 'datatype', 'symbols', 'start', 'end']
    for k in required:
        if k not in job:
            raise ValueError(f"job 缺少必填字段：{k}")
    if job['market'] not in MARKET_MAP:
        raise ValueError(f"未知市场：{job['market']}")
    if job['datatype'] not in DATATYPE_DIR:
        raise ValueError(f"未知数据类型：{job['datatype']}")
    if job['datatype'] == 'klines':
        if 'interval' not in job:
            raise ValueError("klines 任务必须提供 interval")
        iv = str(job['interval'])
        if iv not in VALID_INTERVALS:
            print(f"[WARN] 非常见周期 '{iv}'，如果服务端存在也可用；若失败请换为常见周期：{sorted(VALID_INTERVALS)}")
    # symbols -> list 统一
    if not isinstance(job['symbols'], list):
        job['symbols'] = [job['symbols']]


def main():
    parser = argparse.ArgumentParser(description="Binance Vision 配置驱动下载器")
    parser.add_argument('--config', required=True, help='YAML 配置文件路径')
    args = parser.parse_args()

    cfg = load_config(args.config)
    jobs = cfg.get('jobs', [])
    if not jobs:
        print("配置中未发现 jobs")
        return

    for job in jobs:
        validate_job(job)
        run_job(job)

if __name__ == '__main__':
    main()
