"""
task.conf 解析器
================

平台范式（"算法集成核心.png"）的 getaskconfig 只是 readlines + split('=')，
逻辑很简单，但有几个坑要补：

1. 注释处理：行内有 "#name=xx #description=xx" 这种"系统参数注解"，必须切掉
   再做 = split，否则 value 会粘上一坨注释
2. 类型推断：原 demo 里 args_list[k] = v 都是字符串。但下游用 num_classes/use_xxx
   这类参数时希望是 int / bool / tuple / list（参考 PDF 表）
3. 空白容忍：行首尾空白、引号要去
4. 跨平台路径：value 可能是 Windows 路径，照样原样回传

公开接口：
    load_task_conf(path) -> dict[str, Any]    # 类型推断后的 key->value
    get_task_config(path) -> dict[str, str]   # 兼容 PDF 范式的纯字符串字典
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any, Dict


_INLINE_COMMENT_RE = re.compile(r"\s*#name=")


def _strip_inline_comment(line: str) -> str:
    """切掉  '#name=系统参数 #description=...' 这种行内注解"""
    m = _INLINE_COMMENT_RE.search(line)
    if m:
        return line[: m.start()]
    return line


def _coerce(value: str) -> Any:
    """
    类型推断：
        'True' / 'False'          -> bool
        '123'                     -> int
        '1.5'                     -> float
        "(...)" / "[...]" / "{...}" -> ast.literal_eval
        "'abc'" / '"abc"'         -> 去引号字符串
        'None'                    -> None
        其它                      -> 原样字符串
    """
    v = value.strip()
    if v == "":
        return ""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none" or v == "null":
        return None

    # 带引号的字符串
    if (len(v) >= 2) and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
        return v[1:-1]

    # 容器字面量 (元组/列表/字典)
    if v[0] in "([{" and v[-1] in ")]}":
        try:
            return ast.literal_eval(v)
        except Exception:
            return v  # 解析失败就原样回传，下游自己处理

    # 数字
    if re.fullmatch(r"[-+]?\d+", v):
        try:
            return int(v)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", v) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+", v
    ):
        try:
            return float(v)
        except ValueError:
            pass

    return v


def load_task_conf(path: str) -> Dict[str, Any]:
    """
    解析 task.conf，返回 {key: 类型化 value} 字典。
    文件不存在会抛 FileNotFoundError，由调用方决定怎么处理。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"task.conf 不存在: {os.path.abspath(path)}")

    args: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.strip("\n").strip("\r")
            if not line or line.lstrip().startswith("#"):
                continue
            line = _strip_inline_comment(line)
            line = line.strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().lstrip(" ").rstrip(" ")
            value = value.strip()
            if not key:
                continue
            args[key] = _coerce(value)
    return args


def get_task_config(configfile: str) -> Dict[str, str]:
    """
    兼容"算法集成核心.png"里给的旧接口：返回 str->str 字典。
    内部仍用 load_task_conf 解析，但 value 重新 str 化以保持向后兼容。
    """
    typed = load_task_conf(configfile)
    return {k: ("" if v is None else str(v)) for k, v in typed.items()}


if __name__ == "__main__":
    import json
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "configs/task.conf"
    print(f"loading {os.path.abspath(p)}")
    cfg = load_task_conf(p)
    print(json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
