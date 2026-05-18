"""
后处理：把检测 JSON 结果转 SHP（多边形=外接矩形）
=================================================

输入: <stem>_dets.json （integration.infer_runner 或 deploy.infer_large_image 输出）
      参考 GeoTIFF（用于读取坐标系 / 仿射变换）
输出: <out>.shp 或 <out>.geojson

三档兜底自动选驱动:
    1) fiona  (推荐)
    2) osgeo
    3) 纯 Python 写 GeoJSON

用法:
    python postprocess.py --dets a_dets.json --ref a.tif --out a_dets.shp
    python postprocess.py --dets a_dets.json --out a_dets.geojson    # 不需要坐标系，输出像素坐标
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def _detect_backend():
    try:
        import fiona  # noqa: F401
        return 'fiona'
    except Exception:
        pass
    try:
        from osgeo import ogr, osr  # noqa: F401
        return 'osgeo'
    except Exception:
        pass
    return 'geojson'


def _load_dets(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _bbox_to_polygon_xyxy(bbox_xywh, transform=None):
    """xywh -> Polygon (4 顶点)。transform 给定则做像素->地理换算。"""
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    if transform is not None:
        corners = [transform * c for c in corners]
    return corners


def _write_shp_fiona(polys, out_shp, crs_wkt):
    import fiona
    from fiona.crs import from_wkt
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int', 'score': 'float',
                       'class_id': 'int', 'class': 'str'},
    }
    crs = None
    if crs_wkt:
        try:
            crs = fiona.crs.CRS.from_wkt(crs_wkt)
        except Exception:
            try:
                crs = from_wkt(crs_wkt)
            except Exception:
                crs = None
    with fiona.open(str(out_shp), 'w', driver='ESRI Shapefile',
                    schema=schema, crs=crs, encoding='utf-8') as dst:
        for i, p in enumerate(polys):
            dst.write({
                'geometry': {'type': 'Polygon', 'coordinates': [p['coords']]},
                'properties': {'id': i + 1, 'score': p['score'],
                               'class_id': p['class_id'],
                               'class': p['class']},
            })


def _write_shp_osgeo(polys, out_shp, crs_wkt):
    from osgeo import ogr, osr
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        p = str(out_shp)[:-4] + ext
        if os.path.exists(p):
            os.remove(p)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    ds = drv.CreateDataSource(str(out_shp))
    srs = osr.SpatialReference()
    if crs_wkt:
        srs.ImportFromWkt(crs_wkt)
    layer = ds.CreateLayer('detection', srs, ogr.wkbPolygon)
    for fname, ftype in [('id', ogr.OFTInteger), ('score', ogr.OFTReal),
                         ('class_id', ogr.OFTInteger), ('class', ogr.OFTString)]:
        layer.CreateField(ogr.FieldDefn(fname, ftype))
    for i, p in enumerate(polys):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField('id', i + 1)
        feat.SetField('score', float(p['score']))
        feat.SetField('class_id', int(p['class_id']))
        feat.SetField('class', str(p['class']))
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in p['coords']:
            ring.AddPoint_2D(float(x), float(y))
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
        feat = None
    ds = None


def _write_geojson(polys, out_geojson, crs_wkt):
    features = []
    for i, p in enumerate(polys):
        features.append({
            'type': 'Feature',
            'id': i + 1,
            'properties': {'id': i + 1,
                           'score': float(p['score']),
                           'class_id': int(p['class_id']),
                           'class': str(p['class'])},
            'geometry': {'type': 'Polygon',
                          'coordinates': [p['coords']]},
        })
    fc = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': (crs_wkt[:300] + '...') if len(crs_wkt) > 300 else crs_wkt},
        },
        'features': features,
    }
    with open(out_geojson, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False)


def dets_to_shp(dets_json, out_path, ref_tif=None):
    dets = _load_dets(dets_json)
    if not dets:
        print('[vec] 空检测结果，跳过', flush=True)
        return None

    transform = None
    crs_wkt = ''
    if ref_tif:
        try:
            import rasterio
            with rasterio.open(ref_tif) as src:
                transform = src.transform
                crs = src.crs
                crs_wkt = crs.to_wkt() if crs else ''
        except Exception as e:
            print(f'[vec] 参考栅格读取失败({e})，输出像素坐标', flush=True)

    polys = []
    for d in dets:
        coords = _bbox_to_polygon_xyxy(d['bbox'], transform)
        polys.append({
            'coords': [list(c) for c in coords],
            'score': float(d.get('score', 0.0)),
            'class_id': int(d.get('category_id', 0)),
            'class': str(d.get('category', '')),
        })

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    backend = _detect_backend()
    print(f'[vec] backend = {backend}, polys = {len(polys)}', flush=True)

    if out_p.suffix.lower() == '.shp':
        if backend == 'fiona':
            _write_shp_fiona(polys, out_p, crs_wkt)
            print(f'[ok] shp (fiona) -> {out_p}', flush=True)
        elif backend == 'osgeo':
            _write_shp_osgeo(polys, out_p, crs_wkt)
            print(f'[ok] shp (osgeo) -> {out_p}', flush=True)
        else:
            geo = out_p.with_suffix('.geojson')
            _write_geojson(polys, geo, crs_wkt)
            print(f'[warn] fiona/osgeo 都不可用，改写 GeoJSON: {geo}', flush=True)
            out_p = geo
    else:
        _write_geojson(polys, out_p, crs_wkt)
        print(f'[ok] geojson -> {out_p}', flush=True)
    return str(out_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', required=True, help='检测结果 JSON')
    ap.add_argument('--out', required=True, help='输出 .shp 或 .geojson')
    ap.add_argument('--ref', help='参考 GeoTIFF（用于读取坐标系/仿射）')
    args = ap.parse_args()
    dets_to_shp(args.dets, args.out, args.ref)


if __name__ == '__main__':
    main()
