"""
graph_lib.py — построение графа железнодорожной сети, назначение пропускной
способности (capacity) рёбрам на основе реальных атрибутов путей, расчёт
max-flow/min-cut и симуляция отказов узлов/рёбер.

Используется приложением app.py (Streamlit), но не зависит от Streamlit —
можно импортировать и в Colab/Jupyter напрямую.
"""

import os
import pickle
import logging
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, box
from shapely.ops import nearest_points
from shapely.strtree import STRtree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COORD_PRECISION = 3

DEFAULT_EXCLUDE_RAILWAY_TYPES = {
    'abandoned', 'construction', 'disused', 'proposed', 'razed',
    'dismantled', 'yard', 'service', 'siding', 'spur',
}


def _round_coords(x, y):
    return (round(x, COORD_PRECISION), round(y, COORD_PRECISION))


# ---------------------------------------------------------------------------
# Загрузка и предобработка
# ---------------------------------------------------------------------------

def load_and_preprocess(railways_path, stations_path, target_crs='EPSG:3857',
                         bbox=None, exclude_types=DEFAULT_EXCLUDE_RAILWAY_TYPES,
                         simplify_tolerance=None):
    railways = gpd.read_file(railways_path)
    stations = gpd.read_file(stations_path)

    if railways.crs is None:
        railways = railways.set_crs('EPSG:4326')
    if stations.crs is None:
        stations = stations.set_crs('EPSG:4326')

    if bbox is not None:
        clip_geom = gpd.GeoSeries([box(*bbox)], crs='EPSG:4326').to_crs(railways.crs)
        railways = gpd.clip(railways, clip_geom)
        stations = gpd.clip(stations, clip_geom)

    railways = railways[~(railways.geometry.is_empty | railways.geometry.isna())]
    stations = stations[~(stations.geometry.is_empty | stations.geometry.isna())]

    if exclude_types and 'railway' in railways.columns:
        railways = railways[~railways['railway'].isin(exclude_types)]

    railways = railways.to_crs(target_crs)
    stations = stations.to_crs(target_crs)

    railways = railways[railways.geometry.type.isin(['LineString', 'MultiLineString'])]
    stations = stations[stations.geometry.type == 'Point']

    if simplify_tolerance:
        railways = railways.copy()
        railways['geometry'] = railways.geometry.simplify(simplify_tolerance, preserve_topology=True)

    if len(railways) == 0 or len(stations) == 0:
        raise ValueError("После фильтрации/обрезки не осталось данных.")

    if 'name' not in stations.columns:
        stations['name'] = stations.index.astype(str)
    else:
        missing = stations['name'].isna()
        stations.loc[missing, 'name'] = 'station_' + stations.index[missing].astype(str)
    if stations['name'].duplicated().any():
        dup = stations['name'].duplicated()
        stations.loc[dup, 'name'] = stations.loc[dup, 'name'] + '_' + stations.index[dup].astype(str)

    return railways, stations


# ---------------------------------------------------------------------------
# Назначение пропускной способности (capacity) на основе атрибутов путей
# ---------------------------------------------------------------------------

def _parse_tracks(value):
    """OSM-тег 'tracks' обычно строка вида '2' — парсим безопасно."""
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return None


def estimate_default_capacity(row):
    """
    Эвристика пропускной способности одного оригинального пути (до эффекта
    объединения в общий граф), основанная на реальных OSM-атрибутах:
      - 'tracks' (число путей) — главный сигнал, если есть;
      - electrified — электрифицированные линии обычно магистральные;
      - 'railway' == 'rail' vs 'light_rail'/'narrow_gauge' — снижает капасити.
    Это ОТПРАВНАЯ ТОЧКА. Вы можете переопределить любое значение вручную
    в интерфейсе приложения — они хранятся как overrides поверх этой базы.
    """
    tracks = _parse_tracks(row.get('tracks')) if 'tracks' in row else None
    if tracks is not None:
        base = tracks
    else:
        base = 1
        electrified = str(row.get('electrified', '')).lower()
        if electrified in ('yes', 'contact_line'):
            base = 2

    railway_type = str(row.get('railway', '')).lower()
    if railway_type in ('narrow_gauge', 'light_rail', 'miniature', 'tram'):
        base = max(1, base - 1)

    return float(base)


def assign_segment_capacities(railways, segments, match_buffer=5.0):
    """
    Для каждого сегмента (после union_all/разбиения) находит ближайшую
    ИСХОДНУЮ линию из railways и присваивает эвристическую capacity на
    основе её атрибутов. Приблизительный, но практичный способ перенести
    реальные атрибуты (число путей, электрификация) на топологический граф.
    """
    orig_geoms = list(railways.geometry)
    orig_caps = [estimate_default_capacity(row) for _, row in railways.iterrows()]
    tree = STRtree(orig_geoms)

    seg_capacity = []
    for seg in segments:
        mid = seg.interpolate(0.5, normalized=True)
        try:
            idxs = tree.query(mid, predicate="dwithin", distance=match_buffer)
        except TypeError:
            idxs = tree.query(mid.buffer(match_buffer))
        if len(idxs) == 0:
            seg_capacity.append(1.0)  # запасное значение по умолчанию
            continue
        # берём ближайшую из найденных
        best_idx = min(idxs, key=lambda i: orig_geoms[i].distance(mid))
        seg_capacity.append(orig_caps[best_idx])
    return seg_capacity


# ---------------------------------------------------------------------------
# Построение графа (с capacity на рёбрах)
# ---------------------------------------------------------------------------

def build_graph(railways, stations, snap_threshold=100.0, with_capacity=True):
    all_lines = railways.geometry.union_all()

    if all_lines.geom_type == 'MultiLineString':
        segments = list(all_lines.geoms)
    elif all_lines.geom_type == 'LineString':
        segments = [all_lines]
    else:
        segments = [g for g in all_lines.geoms if g.geom_type == 'LineString']

    if not segments:
        return nx.Graph(), {}

    seg_capacities = assign_segment_capacities(railways, segments) if with_capacity else [1.0] * len(segments)

    seg_tree = STRtree(segments)
    nodes = set()
    station_nodes = {}

    for _, station in stations.iterrows():
        point = station.geometry
        nearest_idx = seg_tree.nearest(point)
        if nearest_idx is None:
            continue
        nearest_seg = segments[nearest_idx]
        nearest_pt = nearest_points(point, nearest_seg)[1]
        dist = point.distance(nearest_pt)
        if dist <= snap_threshold:
            coords = _round_coords(nearest_pt.x, nearest_pt.y)
            nodes.add(coords)
            station_nodes[station['name']] = coords

    for seg in segments:
        c = seg.coords
        nodes.add(_round_coords(c[0][0], c[0][1]))
        nodes.add(_round_coords(c[-1][0], c[-1][1]))

    node_points = [Point(x, y) for (x, y) in nodes]
    node_list = list(nodes)
    node_tree = STRtree(node_points)

    seg_nodes = {}
    for i, seg in enumerate(segments):
        try:
            possible = node_tree.query(seg, predicate="dwithin", distance=0.5)
        except TypeError:
            possible = node_tree.query(seg.buffer(0.5))
        pts = []
        for idx in possible:
            pt = node_points[idx]
            if seg.distance(pt) <= 0.5:
                pts.append((node_list[idx], seg.project(pt)))
        if pts:
            pts.sort(key=lambda x: x[1])
            seg_nodes[i] = pts

    G = nx.Graph()
    for c in nodes:
        G.add_node(c, pos=c)

    for i, pts in seg_nodes.items():
        cap = seg_capacities[i]
        for j in range(len(pts) - 1):
            c1, d1 = pts[j]
            c2, d2 = pts[j + 1]
            length = abs(d2 - d1)
            if length > 0.01:
                G.add_edge(c1, c2, weight=length, capacity=cap, base_capacity=cap)

    for i, seg in enumerate(segments):
        if i not in seg_nodes or len(seg_nodes[i]) < 2:
            c1 = _round_coords(seg.coords[0][0], seg.coords[0][1])
            c2 = _round_coords(seg.coords[-1][0], seg.coords[-1][1])
            if c1 in G.nodes and c2 in G.nodes and c1 != c2 and seg.length > 0.01:
                cap = seg_capacities[i]
                G.add_edge(c1, c2, weight=seg.length, capacity=cap, base_capacity=cap)

    return G, station_nodes


# ---------------------------------------------------------------------------
# Тайлинг + кэш (как в Colab-версии)
# ---------------------------------------------------------------------------

def make_bbox_tiles(full_bbox, n_x=3, n_y=3):
    minx, miny, maxx, maxy = full_bbox
    xs = np.linspace(minx, maxx, n_x + 1)
    ys = np.linspace(miny, maxy, n_y + 1)
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(n_x) for j in range(n_y)]


def build_graph_tiled(railways_path, stations_path, full_bbox, n_x=3, n_y=3,
                       snap_threshold=100.0, exclude_types=DEFAULT_EXCLUDE_RAILWAY_TYPES,
                       simplify_tolerance=3.0, target_crs='EPSG:3857', progress_cb=None):
    tiles = make_bbox_tiles(full_bbox, n_x, n_y)
    graphs = []
    all_station_nodes = {}

    for k, bbox in enumerate(tiles):
        if progress_cb:
            progress_cb(k + 1, len(tiles))
        try:
            railways, stations = load_and_preprocess(
                railways_path, stations_path, target_crs=target_crs, bbox=bbox,
                exclude_types=exclude_types, simplify_tolerance=simplify_tolerance,
            )
        except ValueError:
            continue
        G_tile, stations_tile = build_graph(railways, stations, snap_threshold=snap_threshold)
        graphs.append(G_tile)
        all_station_nodes.update(stations_tile)

    if not graphs:
        raise ValueError("Ни один тайл не дал данных.")

    G = nx.compose_all(graphs)
    return G, all_station_nodes


def save_graph_cache(G, station_nodes, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump({'graph': G, 'station_nodes': station_nodes}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_cache(cache_path):
    if not os.path.exists(cache_path):
        return None, None
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data['graph'], data['station_nodes']


def get_or_build_graph(railways_path, stations_path, cache_path, full_bbox=None,
                        n_x=3, n_y=3, force_rebuild=False, progress_cb=None, **kwargs):
    if not force_rebuild:
        G, station_nodes = load_graph_cache(cache_path)
        if G is not None:
            return G, station_nodes

    if full_bbox is not None:
        G, station_nodes = build_graph_tiled(railways_path, stations_path, full_bbox, n_x, n_y,
                                              progress_cb=progress_cb, **kwargs)
    else:
        railways, stations = load_and_preprocess(railways_path, stations_path)
        G, station_nodes = build_graph(railways, stations)

    save_graph_cache(G, station_nodes, cache_path)
    return G, station_nodes


# ---------------------------------------------------------------------------
# Базовые запросы: путь, статистика
# ---------------------------------------------------------------------------

def shortest_path(G, station_nodes, start_name, end_name, weight='weight'):
    if start_name not in station_nodes or end_name not in station_nodes:
        raise ValueError("Станция не найдена.")
    s, t = station_nodes[start_name], station_nodes[end_name]
    if s not in G or t not in G:
        return None, None
    try:
        path = nx.shortest_path(G, s, t, weight=weight)
        length = nx.shortest_path_length(G, s, t, weight=weight)
        return path, length
    except nx.NetworkXNoPath:
        return None, None


def network_statistics(G, betweenness_sample_threshold=1000, betweenness_k=200, seed=42):
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    components = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in components), reverse=True)
    degrees = dict(G.degree())
    avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0

    if n_nodes > betweenness_sample_threshold:
        betweenness = nx.betweenness_centrality(G, k=min(betweenness_k, n_nodes), weight='weight', seed=seed)
    else:
        betweenness = nx.betweenness_centrality(G, weight='weight')
    top = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        'n_nodes': n_nodes, 'n_edges': n_edges, 'n_components': len(sizes),
        'component_sizes': sizes, 'avg_degree': avg_degree, 'top_betweenness': top,
    }


# ---------------------------------------------------------------------------
# MAX-FLOW / MIN-CUT
# ---------------------------------------------------------------------------

def max_flow_between(G, source, target, capacity_attr='capacity'):
    """
    Полноценный max-flow (алгоритм Диница по умолчанию в networkx) между
    двумя узлами по атрибуту 'capacity' рёбер.
    Возвращает: (flow_value, flow_dict, cut_edges, reachable_from_source)
    """
    if source not in G or target not in G:
        raise ValueError("Узел-источник или узел-сток отсутствует в графе.")
    if not nx.has_path(G, source, target):
        return 0.0, {}, [], set()

    flow_value, flow_dict = nx.maximum_flow(G, source, target, capacity=capacity_attr)
    cut_value, (reachable, non_reachable) = nx.minimum_cut(G, source, target, capacity=capacity_attr)

    cut_edges = []
    for u in reachable:
        for v in G.neighbors(u):
            if v in non_reachable:
                cut_edges.append((u, v))

    return flow_value, flow_dict, cut_edges, reachable


def apply_capacity_overrides(G, overrides):
    """
    overrides: dict {(u, v) или frozenset({u, v}): new_capacity}
    Возвращает КОПИЮ графа с применёнными переопределениями (базовый граф
    не трогаем — это важно, чтобы можно было экспериментировать и откатываться).
    """
    G2 = G.copy()
    for key, cap in overrides.items():
        u, v = tuple(key) if not isinstance(key, tuple) else key
        if G2.has_edge(u, v):
            G2[u][v]['capacity'] = float(cap)
    return G2


# ---------------------------------------------------------------------------
# Симуляция отказов
# ---------------------------------------------------------------------------

def simulate_failures(G, failed_nodes=None, failed_edges=None):
    """
    Возвращает КОПИЮ графа с удалёнными узлами/рёбрами (не мутирует исходный).
    failed_edges: список (u, v) — порядок не важен, ищем в обе стороны.
    """
    G2 = G.copy()
    if failed_nodes:
        G2.remove_nodes_from([n for n in failed_nodes if n in G2])
    if failed_edges:
        for u, v in failed_edges:
            if G2.has_edge(u, v):
                G2.remove_edge(u, v)
    return G2


def compare_before_after(G_base, G_after, station_nodes, start_name, end_name, capacity_attr='capacity'):
    """
    Сравнение сценария до/после отказа: длина кратчайшего пути и max-flow.
    Удобная функция для сводки в UI.
    """
    s, t = station_nodes[start_name], station_nodes[end_name]

    result = {}
    for label, G in (('before', G_base), ('after', G_after)):
        if s not in G or t not in G or not nx.has_path(G, s, t):
            result[label] = {'connected': False, 'path_length_km': None, 'max_flow': 0.0}
            continue
        length = nx.shortest_path_length(G, s, t, weight='weight')
        flow, _, _, _ = max_flow_between(G, s, t, capacity_attr=capacity_attr)
        result[label] = {'connected': True, 'path_length_km': length / 1000.0, 'max_flow': flow}

    return result
