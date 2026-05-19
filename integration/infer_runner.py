"""
推理核心（三栈统一：pth / onnx / om）
=====================================

按文档约定，推理时参数从 clie_lib/configs/task.conf 读：
    必填：
        load_model_path     模型路径（.pth / .onnx / .om）
        img_list_file       影像列表（每行一个绝对路径），单时相
        output_root         输出目录
    可选：
        bootstrap_servers / topic / projectId / taskId   Kafka 上报
        use_color_out       输出可视化 PNG（画框）
        use_shapefile_out   输出 SHP（GeoTIFF 输入才有意义）
        num_classes / classes_name / palette   类别信息
        score_thr           置信度阈值，默认 0.3
        iou_thr             NMS IoU 阈值，默认 0.5
        max_per_img         每张图最多框数，默认 100
        device              强制 device（cuda/npu/cpu/om）
        base_config         pth 推理时的 mmdet config（不传则用默认）
        color_table_file    类别 -> 颜色映射 txt（"idx,name,r,g,b" 每行）
        infer_size          ONNX/OM 模型 input HxW，例如 (800,800)；pth 自动决定

设备/后端选择：
    load_model_path 后缀决定走哪个 runner：
        .om   -> NPU pyACL
        .onnx -> onnxruntime
        .pth  -> PyTorch（GPU 推理，NPU 也行，看 device）

输出格式：
    每张图输出：
        <stem>_dets.json   COCO 风格 [{bbox:[x,y,w,h], score, category_id, ...}, ...]
        <stem>_vis.png     可选，画框可视化
        <stem>_dets.shp    可选，仅 GeoTIFF 输入；多边形=外接矩形，属性带 class/score
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============= 复用 deploy 里的 Kafka 发送 =============
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY = os.path.join(os.path.dirname(_HERE), "deploy")
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)

try:
    from send_kafka_msg import send as _kafka_send_env  # type: ignore  # noqa: E402
except Exception:
    def _kafka_send_env(progress, status, info):
        print(f"[Log Only] {progress}% - {status} - {info}", flush=True)

from .conf_parser import load_task_conf
from .config_builder import _read_floats_file
from .device_utils import detect_device, setup_device_env


# ============================================================
# PyTorch (pth) 推理 runner —— 给 GPU / NPU 直接吃 pth 用
# ============================================================
class PthRunner:
    """
    直接吃 mmdet ckpt（.pth + 对应 config）做检测推理。

    用法：
        runner = PthRunner('xxx.pth',
                           config='configs/retinanet_r50_fpn_object.py',
                           device='cuda',
                           score_thr=0.3)

    输入: image: numpy.ndarray (H, W, 3) BGR uint8（OpenCV 读出来直接给）
    输出: dict with keys:
        'bboxes': (N, 4) float32, xyxy
        'scores': (N,)   float32
        'labels': (N,)   int64
    """

    def __init__(
        self,
        ckpt: str,
        config: str,
        device: str = "cuda",
        score_thr: float = 0.3,
        task_cfg: Optional[dict] = None,
    ):
        import torch
        from mmengine.config import Config
        from mmdet.apis import init_detector, inference_detector

        self.torch = torch
        self.inference_detector = inference_detector

        cfg = Config.fromfile(config)
        # SyncBN 单进程跑不动，强制 BN
        for k in ("backbone", "neck"):
            sub = cfg.model.get(k)
            if isinstance(sub, dict) and "norm_cfg" in sub:
                sub["norm_cfg"] = dict(type="BN", requires_grad=True)
        if "roi_head" in cfg.model and isinstance(cfg.model.roi_head, dict):
            bb = cfg.model.roi_head.get("bbox_head")
            if isinstance(bb, dict) and "norm_cfg" in bb:
                bb["norm_cfg"] = dict(type="BN", requires_grad=True)

        # 类别同步
        task_cfg = task_cfg or {}
        nc_task = task_cfg.get("num_classes")
        if nc_task is not None:
            from .config_builder import _patch_detector_classes
            try:
                _patch_detector_classes(cfg, int(nc_task))
            except Exception as e:
                print(f"[PthRunner] num_classes 同步失败({e})，沿用 base config", flush=True)

        # mean/std
        mean_from_file = _read_floats_file(task_cfg.get("mean_file", ""), None)
        std_from_file = _read_floats_file(task_cfg.get("std_file", ""), None)
        dp = cfg.model.get("data_preprocessor", {}) or {}
        if mean_from_file:
            dp["mean"] = mean_from_file
        if std_from_file:
            dp["std"] = std_from_file
        cfg.model["data_preprocessor"] = dp

        torch_device = "cuda:0" if device == "cuda" else (
            "npu:0" if device == "npu" else "cpu"
        )
        self.model = init_detector(cfg, ckpt, device=torch_device)
        self.model.eval()
        self.device = torch_device
        self.score_thr = float(score_thr)

    @property
    def num_classes(self) -> int:
        # 兼容单/两阶段
        head = getattr(self.model, "bbox_head", None)
        if head is not None and hasattr(head, "num_classes"):
            return int(head.num_classes)
        roi = getattr(self.model, "roi_head", None)
        if roi is not None:
            bh = getattr(roi, "bbox_head", None)
            if bh is not None and hasattr(bh, "num_classes"):
                return int(bh.num_classes)
        return 80  # COCO 默认

    def infer(self, img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        result = self.inference_detector(self.model, img_bgr)
        # mmdet 3.x: result.pred_instances 含 bboxes / scores / labels
        pi = result.pred_instances
        bboxes = pi.bboxes.detach().cpu().numpy().astype(np.float32)  # (N,4) xyxy
        scores = pi.scores.detach().cpu().numpy().astype(np.float32)
        labels = pi.labels.detach().cpu().numpy().astype(np.int64)
        keep = scores >= self.score_thr
        return dict(
            bboxes=bboxes[keep],
            scores=scores[keep],
            labels=labels[keep],
        )

    def close(self):
        del self.model


# ============================================================
# ONNXRunner / OMRunner
# ============================================================
# ONNX/OM 模型约定（与 tools/export_onnx.py 导出的一致）：
#   input:   'input', shape (1, 3, H, W), float32, [0,255], RGB
#   outputs:
#       'dets':   (1, max_dets, 5)  -> [x1, y1, x2, y2, score]
#       'labels': (1, max_dets)     -> int64
#   有效检测数量由 score==0 截断（导出时已 pad）
# ============================================================
class ONNXRunner:
    """ONNX 检测推理（CPU/GPU 回落）"""

    def __init__(self, onnx_path: str, infer_size: Tuple[int, int] = (800, 800)):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.infer_h, self.infer_w = infer_size
        print(f"[ONNXRunner] in={self.in_name} outs={self.out_names} "
              f"size=({self.infer_h},{self.infer_w})", flush=True)

    def infer(self, img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        import cv2

        h0, w0 = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.infer_w, self.infer_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1).astype(np.float32)[None]  # (1,3,H,W) [0,255]
        outs = self.sess.run(self.out_names, {self.in_name: x})
        return _decode_dets(outs, (h0, w0), (self.infer_h, self.infer_w))

    def close(self):
        pass


class OMRunner:
    """pyACL 推理（昇腾 NPU）"""

    def __init__(self, om_path: str, device_id: int = 0,
                 infer_size: Tuple[int, int] = (800, 800)):
        import acl
        self._acl = acl
        ret = acl.init(); self._chk(ret, "acl.init")
        ret = acl.rt.set_device(device_id); self._chk(ret, "set_device")
        self.ctx, ret = acl.rt.create_context(device_id); self._chk(ret, "create_context")
        self.model_id, ret = acl.mdl.load_from_file(om_path); self._chk(ret, "load_model")
        self.model_desc = acl.mdl.create_desc()
        self._chk(acl.mdl.get_desc(self.model_desc, self.model_id), "get_desc")

        # 缓存 IO 大小
        self.n_inputs = acl.mdl.get_num_inputs(self.model_desc)
        self.n_outputs = acl.mdl.get_num_outputs(self.model_desc)
        self.input_sizes = [acl.mdl.get_input_size_by_index(self.model_desc, i)
                            for i in range(self.n_inputs)]
        self.output_sizes = [acl.mdl.get_output_size_by_index(self.model_desc, i)
                             for i in range(self.n_outputs)]
        self.infer_h, self.infer_w = infer_size
        print(f"[OMRunner] {om_path} n_in={self.n_inputs} n_out={self.n_outputs} "
              f"in_sizes={self.input_sizes} out_sizes={self.output_sizes} "
              f"size=({self.infer_h},{self.infer_w})", flush=True)

    @staticmethod
    def _chk(ret, msg):
        if ret != 0:
            raise RuntimeError(f"{msg} failed, ret={ret}")

    def infer(self, img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        import cv2
        acl = self._acl

        h0, w0 = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.infer_w, self.infer_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32)[None])
        # 单输入
        assert self.n_inputs == 1
        in_dev, ret = acl.rt.malloc(self.input_sizes[0], 0); self._chk(ret, "malloc in")
        acl.rt.memcpy(in_dev, self.input_sizes[0], x.ctypes.data, x.nbytes, 1)
        in_buf = acl.create_data_buffer(in_dev, self.input_sizes[0])
        in_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(in_ds, in_buf)

        # 多输出
        out_devs, out_bufs = [], []
        out_ds = acl.mdl.create_dataset()
        for i in range(self.n_outputs):
            dev, ret = acl.rt.malloc(self.output_sizes[i], 0); self._chk(ret, "malloc out")
            out_devs.append(dev)
            buf = acl.create_data_buffer(dev, self.output_sizes[i])
            out_bufs.append(buf)
            acl.mdl.add_dataset_buffer(out_ds, buf)

        self._chk(acl.mdl.execute(self.model_id, in_ds, out_ds), "execute")

        # 拷回 host —— 这里 dtype 写死 float32 / int64 是按 export_onnx.py 的约定。
        # 第 0 个输出 dets (float32)，第 1 个输出 labels (int64)
        outs = []
        # dets
        dets = np.zeros(self.output_sizes[0] // 4, dtype=np.float32)
        acl.rt.memcpy(dets.ctypes.data, dets.nbytes, out_devs[0], self.output_sizes[0], 2)
        outs.append(dets)
        # labels
        if self.n_outputs >= 2:
            labels = np.zeros(self.output_sizes[1] // 8, dtype=np.int64)
            acl.rt.memcpy(labels.ctypes.data, labels.nbytes, out_devs[1], self.output_sizes[1], 2)
            outs.append(labels)

        # release
        acl.destroy_data_buffer(in_buf)
        for b in out_bufs:
            acl.destroy_data_buffer(b)
        acl.mdl.destroy_dataset(in_ds)
        acl.mdl.destroy_dataset(out_ds)
        acl.rt.free(in_dev)
        for d in out_devs:
            acl.rt.free(d)

        # reshape & decode
        # dets: (1, max_dets, 5); labels: (1, max_dets)
        # max_dets 通过 output_size 倒推
        max_dets = self.output_sizes[0] // (4 * 5)
        dets = outs[0].reshape(1, max_dets, 5)
        if len(outs) >= 2:
            labels = outs[1].reshape(1, max_dets)
        else:
            labels = np.zeros((1, max_dets), dtype=np.int64)

        return _decode_dets([dets, labels], (h0, w0), (self.infer_h, self.infer_w))

    def close(self):
        acl = self._acl
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.model_desc)
        acl.rt.destroy_context(self.ctx)
        acl.rt.reset_device(0)
        acl.finalize()


def _decode_dets(outs, orig_hw: Tuple[int, int],
                 infer_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    ONNX/OM 输出 -> 统一字典。
    outs: [dets (1,N,5), labels (1,N)]
    把 bbox 从 infer 尺度还原回原图尺度，并 drop score<=0 的 pad 项。
    """
    dets = outs[0]
    labels = outs[1] if len(outs) > 1 else np.zeros(dets.shape[:2], dtype=np.int64)
    if dets.ndim == 3:
        dets = dets[0]
        labels = labels[0]
    # (N, 5) xyxy + score
    scores = dets[:, 4]
    keep = scores > 0
    dets = dets[keep]
    labels = labels[keep]

    # 还原到原图坐标
    h0, w0 = orig_hw
    ih, iw = infer_hw
    sx = w0 / float(iw)
    sy = h0 / float(ih)
    bboxes = dets[:, :4].copy()
    bboxes[:, 0::2] *= sx
    bboxes[:, 1::2] *= sy

    return dict(
        bboxes=bboxes.astype(np.float32),
        scores=dets[:, 4].astype(np.float32),
        labels=labels.astype(np.int64),
    )


def make_runner_any(model_path: str, device: str,
                    base_config: Optional[str] = None,
                    score_thr: float = 0.3,
                    task_cfg: Optional[dict] = None,
                    infer_size: Tuple[int, int] = (800, 800)):
    """根据后缀和设备选 runner"""
    suffix = Path(model_path).suffix.lower()
    if suffix == ".om":
        print(f"[runner] OM (NPU): {model_path}", flush=True)
        return OMRunner(model_path, infer_size=infer_size)
    if suffix == ".onnx":
        print(f"[runner] ONNX ({device}): {model_path}", flush=True)
        return ONNXRunner(model_path, infer_size=infer_size)
    if suffix in (".pth", ".pt"):
        if base_config is None:
            raise ValueError("pth 推理需要 base_config（mmdet config 路径）")
        print(f"[runner] Pth ({device}): {model_path}  cfg={base_config}", flush=True)
        return PthRunner(model_path, base_config, device=device,
                         score_thr=score_thr, task_cfg=task_cfg)
    raise ValueError(f"不识别的模型后缀: {suffix}")


# ============================================================
# 工具
# ============================================================
def _read_list(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"列表文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _is_geotiff(path: str) -> bool:
    return Path(path).suffix.lower() in (".tif", ".tiff")


def _parse_color_table(path: Optional[str], num_classes: int,
                       classes: Optional[List[str]] = None
                       ) -> Tuple[List[str], List[List[int]]]:
    """
    color_table.txt 格式（一行一类）：
        0,car,255,0,0
        1,truck,0,255,0
        ...
    返回 (names, palette[(r,g,b)*n])
    """
    if not path or not os.path.exists(path):
        names = list(classes) if classes else [f"cls_{i}" for i in range(num_classes)]
        # 默认调色板
        rng = np.random.default_rng(seed=42)
        palette = rng.integers(0, 255, size=(num_classes, 3)).tolist()
        return names, palette

    names: List[Optional[str]] = [None] * num_classes
    palette: List[Optional[List[int]]] = [None] * num_classes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
            try:
                idx = int(parts[0])
            except (ValueError, IndexError):
                continue
            if idx >= num_classes:
                continue
            name = parts[1] if len(parts) >= 2 else f"cls_{idx}"
            try:
                rgb = [int(parts[2]), int(parts[3]), int(parts[4])]
            except (ValueError, IndexError):
                rgb = [128, 128, 128]
            names[idx] = name
            palette[idx] = rgb

    # 兜底
    for i in range(num_classes):
        if names[i] is None:
            names[i] = (classes[i] if classes and i < len(classes) else f"cls_{i}")
        if palette[i] is None:
            palette[i] = [128, 128, 128]
    return names, palette  # type: ignore


def _draw_dets(img_bgr: np.ndarray, dets: Dict[str, np.ndarray],
               names: List[str], palette: List[List[int]]) -> np.ndarray:
    import cv2
    out = img_bgr.copy()
    for box, score, lab in zip(dets["bboxes"], dets["scores"], dets["labels"]):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color_rgb = palette[int(lab) % len(palette)]
        # OpenCV BGR
        color = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names[int(lab) % len(names)]} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _imread_utf8(path: str) -> Optional[np.ndarray]:
    """跨平台中文路径友好的 cv2.imread"""
    import cv2
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def _imwrite_utf8(out_path: str, img_arr: np.ndarray, ext: str = ".png") -> None:
    import cv2
    ok, buf = cv2.imencode(ext, img_arr)
    if not ok:
        raise RuntimeError(f"cv2.imencode 失败: {out_path}")
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())


def _dets_to_json(dets: Dict[str, np.ndarray], img_path: str,
                  names: List[str], geotransform=None) -> List[dict]:
    """COCO-like 输出: bbox=[x,y,w,h], score, category_id, category, image_path"""
    out = []
    for box, score, lab in zip(dets["bboxes"], dets["scores"], dets["labels"]):
        x1, y1, x2, y2 = [float(v) for v in box]
        entry = dict(
            image_path=img_path,
            category_id=int(lab),
            category=names[int(lab) % len(names)],
            bbox=[x1, y1, x2 - x1, y2 - y1],
            score=float(score),
        )
        out.append(entry)
    return out


def _save_shp(dets: Dict[str, np.ndarray], img_path: str, out_shp: str,
              names: List[str]) -> Optional[str]:
    """把检测框写成 GeoTIFF 同坐标系下的矩形多边形。"""
    try:
        import rasterio
        from shapely.geometry import box, mapping
    except ImportError:
        print("[shp] rasterio/shapely 未安装，跳过 SHP 输出", flush=True)
        return None

    with rasterio.open(img_path) as src:
        transform = src.transform
        crs = src.crs

    polys = []
    for b, s, lab in zip(dets["bboxes"], dets["scores"], dets["labels"]):
        x1, y1, x2, y2 = b
        # 像素 -> 地理坐标
        gx1, gy1 = transform * (x1, y1)
        gx2, gy2 = transform * (x2, y2)
        polys.append((box(min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)),
                      float(s), int(lab)))

    if not polys:
        return None

    # 优先 fiona，缺就 osgeo
    crs_wkt = crs.to_wkt() if crs else ""
    try:
        import fiona
        schema = {
            "geometry": "Polygon",
            "properties": {"id": "int", "score": "float", "class_id": "int",
                           "class": "str"},
        }
        ff_crs = None
        if crs_wkt:
            try:
                ff_crs = fiona.crs.CRS.from_wkt(crs_wkt)
            except Exception:
                ff_crs = None
        with fiona.open(str(out_shp), "w", driver="ESRI Shapefile",
                        schema=schema, crs=ff_crs, encoding="utf-8") as dst:
            for i, (p, s, lab) in enumerate(polys):
                dst.write({
                    "geometry": mapping(p),
                    "properties": {
                        "id": i + 1, "score": s,
                        "class_id": lab,
                        "class": names[lab % len(names)],
                    },
                })
        return out_shp
    except ImportError:
        pass

    try:
        from osgeo import ogr, osr
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = str(out_shp)[:-4] + ext
            if os.path.exists(p):
                os.remove(p)
        drv = ogr.GetDriverByName("ESRI Shapefile")
        ds = drv.CreateDataSource(str(out_shp))
        srs = osr.SpatialReference()
        if crs_wkt:
            srs.ImportFromWkt(crs_wkt)
        layer = ds.CreateLayer("detection", srs, ogr.wkbPolygon)
        for fname, ftype in [("id", ogr.OFTInteger), ("score", ogr.OFTReal),
                             ("class_id", ogr.OFTInteger), ("class", ogr.OFTString)]:
            layer.CreateField(ogr.FieldDefn(fname, ftype))
        for i, (p, s, lab) in enumerate(polys):
            feat = ogr.Feature(layer.GetLayerDefn())
            feat.SetField("id", i + 1)
            feat.SetField("score", s)
            feat.SetField("class_id", lab)
            feat.SetField("class", names[lab % len(names)])
            feat.SetGeometry(ogr.CreateGeometryFromWkb(p.wkb))
            layer.CreateFeature(feat)
            feat = None
        ds = None
        return out_shp
    except ImportError:
        print("[shp] 既无 fiona 也无 osgeo，SHP 输出跳过", flush=True)
        return None


# ============================================================
# 单图处理
# ============================================================
def _process_one(img_path: str, output_root: str, runner,
                 task_cfg: dict, names: List[str], palette: List[List[int]]) -> dict:
    out = {"json": None, "vis": None, "shp": None}
    stem = Path(img_path).stem
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    img_bgr = _imread_utf8(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"读不到影像: {img_path}")
    if img_bgr.ndim == 2:
        img_bgr = np.stack([img_bgr] * 3, axis=-1)
    if img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    # 大图自动切滑窗：mmdet 默认 test_pipeline 会把整图 Resize 到 (1333, 800)，
    # 一张 7800x7900 的遥感图缩到 ~1333x1350，桥梁/小目标会被压成几像素，根本检不到。
    # 解决：边长超过 tile_size * slide_min_ratio 时，切瓦片单独推理，跨瓦片 NMS。
    h, w = img_bgr.shape[:2]
    tile_size = int(task_cfg.get("tile_size", 800))
    tile_stride = int(task_cfg.get("tile_stride", 600))
    iou_thr = float(task_cfg.get("iou_thr", 0.5))
    slide_min_ratio = float(task_cfg.get("slide_min_ratio", 1.5))
    force_slide = bool(task_cfg.get("use_slide_window", False))
    use_slide = force_slide or (max(h, w) > tile_size * slide_min_ratio)

    if use_slide:
        from deploy.infer_large_image import slide_detect
        import cv2

        print(f"[infer] sliding window: image={h}x{w} "
              f"tile={tile_size} stride={tile_stride} iou_thr={iou_thr}",
              flush=True)
        # slide_detect 要 (3, H, W) RGB uint8
        rgb_chw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        # PthRunner / ONNXRunner / OMRunner 内部都已做过 score_thr 过滤，
        # 这里再传一次给 slide_detect 也无妨（同值则等价 no-op）。
        score_thr_for_slide = float(task_cfg.get("score_thr", 0.3))
        dets = slide_detect(rgb_chw, runner,
                            tile=tile_size, stride=tile_stride,
                            score_thr=score_thr_for_slide, iou_thr=iou_thr)
    else:
        dets = runner.infer(img_bgr)

    # JSON
    js = _dets_to_json(dets, img_path, names)
    js_path = output_root / f"{stem}_dets.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    out["json"] = str(js_path)

    # 可视化
    if bool(task_cfg.get("use_color_out", False)):
        vis = _draw_dets(img_bgr, dets, names, palette)
        vis_path = output_root / f"{stem}_vis.png"
        _imwrite_utf8(str(vis_path), vis, ext=".png")
        out["vis"] = str(vis_path)

    # SHP（GeoTIFF 才有意义）
    if bool(task_cfg.get("use_shapefile_out", False)) and _is_geotiff(img_path):
        shp_path = output_root / f"{stem}_dets.shp"
        out["shp"] = _save_shp(dets, img_path, str(shp_path), names)

    return out


# ============================================================
# Kafka
# ============================================================
def _kafka_send(progress: int, status: str, info: str, task_cfg: dict):
    bs = task_cfg.get("bootstrap_servers")
    topic = task_cfg.get("topic")
    task_id = task_cfg.get("taskId")
    if bs and topic:
        os.environ["KAFKA_SERVER_IP_PORT"] = str(bs)
        os.environ["KAFKA_TOPIC"] = str(topic)
        if task_id is not None:
            os.environ["KAFKA_TASK_ID"] = str(task_id)
    _kafka_send_env(progress, status, info)


# ============================================================
# 主入口
# ============================================================
def run_inference(
    task_conf_path: str = "clie_lib/configs/task.conf",
    device: Optional[str] = None,
    base_config: Optional[str] = None,
):
    os.environ.setdefault("GDAL_FILENAME_IS_UTF8", "YES")
    os.environ.setdefault("SHAPE_ENCODING", "UTF-8")

    print(f"[run_inference] task_conf = {os.path.abspath(task_conf_path)}", flush=True)
    task_cfg = load_task_conf(task_conf_path)
    print(f"[run_inference] keys = {sorted(task_cfg.keys())}", flush=True)

    model_path = task_cfg.get("load_model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"load_model_path 不存在: {model_path}")

    img_list_file = task_cfg.get("img_list_file")
    if not img_list_file:
        raise ValueError("img_list_file 未配置")
    images = _read_list(img_list_file)
    if not images:
        raise ValueError(f"img_list_file 是空的: {img_list_file}")

    output_root = task_cfg.get("output_root")
    if not output_root:
        raise ValueError("output_root 未配置")

    # base_config: pth 必须，om/onnx 不需要
    suffix = Path(model_path).suffix.lower()
    if suffix in (".pth", ".pt") and not base_config:
        base_config = (
            task_cfg.get("base_config")
            or "configs/retinanet_r50_fpn_object.py"
        )

    device = setup_device_env(device or task_cfg.get("device") or detect_device())
    if suffix == ".om" and device != "npu":
        print(f"[warn] 模型是 .om 但 device={device}，强制切到 npu", flush=True)
        device = "npu"

    score_thr = float(task_cfg.get("score_thr", 0.3))
    # ONNX/OM 的输入尺寸
    infer_size_raw = task_cfg.get("infer_size", (800, 800))
    if isinstance(infer_size_raw, str):
        try:
            infer_size_raw = eval(infer_size_raw)
        except Exception:
            infer_size_raw = (800, 800)
    if isinstance(infer_size_raw, (list, tuple)) and len(infer_size_raw) == 2:
        infer_size = (int(infer_size_raw[0]), int(infer_size_raw[1]))
    else:
        infer_size = (800, 800)

    runner = make_runner_any(model_path, device,
                             base_config=base_config,
                             score_thr=score_thr,
                             task_cfg=task_cfg,
                             infer_size=infer_size)

    # 类别/调色板
    classes_name = task_cfg.get("classes_name")
    if isinstance(classes_name, (list, tuple)):
        classes_list = list(classes_name)
    elif isinstance(classes_name, str):
        classes_list = [s.strip().strip("'\"") for s in
                        classes_name.strip("()[] ").split(",") if s.strip()]
    else:
        classes_list = None
    num_classes = int(task_cfg.get("num_classes") or
                       (len(classes_list) if classes_list else 80))
    names, palette = _parse_color_table(task_cfg.get("color_table_file"),
                                         num_classes, classes_list)
    print(f"[run_inference] num_classes={num_classes}, names={names[:5]}{'...' if len(names)>5 else ''}",
          flush=True)

    _kafka_send(0, "running", "推理任务启动", task_cfg)

    t0 = time.time()
    try:
        total = len(images)
        for i, img in enumerate(images, 1):
            print(f"\n===== [{i}/{total}] {img} =====", flush=True)
            try:
                res = _process_one(img, output_root, runner, task_cfg, names, palette)
                print(f"  -> {res}", flush=True)
                # 进度（保留首尾 5%）
                prog = int(5 + (i / total) * 90)
                _kafka_send(min(prog, 95), "running",
                            f"[{i}/{total}] 完成", task_cfg)
            except Exception as e:
                print(f"[error] {img} 推理失败: {e}", flush=True)
                _kafka_send(int(i / total * 95), "running",
                            f"[{i}/{total}] 失败: {e}", task_cfg)
                continue

        elapsed = time.time() - t0
        _kafka_send(100, "completed",
                    f"完成 {total} 张影像推理，耗时 {elapsed:.1f}s", task_cfg)
    finally:
        runner.close()


def main_cli(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser("mmdet predict (integration)")
    parser.add_argument("action", nargs="?", default="infer", choices=["infer"])
    parser.add_argument("--custom-config", default="task",
                        help="文档要求传 'task'，对应 clie_lib/configs/task.conf；"
                        "也可以传文件绝对路径")
    parser.add_argument("--device", default=None,
                        choices=[None, "cuda", "npu", "cpu"])
    parser.add_argument("--base", default=None,
                        help="pth 推理时的 mmdet config（不传则用 configs/retinanet_r50_fpn_object.py）")
    args = parser.parse_args(argv)

    cc = args.custom_config
    if cc == "task" or cc is None:
        task_conf_path = "clie_lib/configs/task.conf"
    elif os.path.isabs(cc) or cc.endswith(".conf"):
        task_conf_path = cc
    else:
        task_conf_path = f"clie_lib/configs/{cc}.conf"

    run_inference(task_conf_path=task_conf_path,
                  device=args.device,
                  base_config=args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
