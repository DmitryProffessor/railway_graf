"""
Streamlit-приложение для интерактивной работы с графом железнодорожной сети:
  - карта сети (folium) с станциями;
  - расчёт кратчайшего пути и MAX-FLOW (пропускная способность) между двумя
    станциями, с реальными весами capacity, оцененными по атрибутам путей
    (число путей / электрификация), которые можно переопределять вручную;
  - симуляция отказов узлов (станций) и рёбер (участков) с сравнением
    "до / после" по длине маршрута и величине max-flow, подсветка min-cut.

Запуск локально:
    pip install -r requirements.txt
    streamlit run app.py

Первый запуск строит граф (может занять время для всей страны) и сохраняет
кэш в graph_cache.pkl рядом с приложением — повторные запуски грузят кэш
мгновенно, если не менять параметры фильтрации/упрощения.
"""

import os
import pickle
import streamlit as st
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
import networkx as nx

import graph_lib as gl

st.set_page_config(page_title="Железнодорожная сеть — симулятор", layout="wide")

# ---------------------------------------------------------------------------
# Координатные преобразования: граф хранится в EPSG:3857 (метры),
# folium рисует в WGS84 (lat/lon).
# ---------------------------------------------------------------------------
TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def xy_to_latlon(x, y):
    lon, lat = TO_WGS84.transform(x, y)
    return lat, lon


def latlon_to_xy(lat, lon):
    x, y = TO_MERCATOR.transform(lon, lat)
    return x, y


# ---------------------------------------------------------------------------
# Состояние сессии
# ---------------------------------------------------------------------------
for key, default in [
    ('G_base', None), ('station_nodes', None),
    ('capacity_overrides', {}), ('failed_nodes', set()), ('failed_edges', set()),
    ('selected_edge', None), ('last_result', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def working_graph():
    """Базовый граф + применённые переопределения capacity (без отказов)."""
    G = gl.apply_capacity_overrides(st.session_state.G_base, st.session_state.capacity_overrides)
    return G


def graph_with_failures():
    """working_graph() + удалённые узлы/рёбра (текущий сценарий отказа)."""
    G = working_graph()
    return gl.simulate_failures(G, st.session_state.failed_nodes, st.session_state.failed_edges)


# ---------------------------------------------------------------------------
# Сайдбар: загрузка/построение графа
# ---------------------------------------------------------------------------
st.sidebar.header("1. Данные и построение графа")

railways_path = st.sidebar.text_input("Путь к railways.geojson", "railways.geojson")
stations_path = st.sidebar.text_input("Путь к stations.geojson", "stations.geojson")
cache_path = st.sidebar.text_input("Файл кэша графа", "graph_cache.pkl")

with st.sidebar.expander("Параметры построения (для тайлинга всей страны)"):
    use_tiling = st.checkbox("Строить по тайлам (для большой территории)", value=False)
    bbox_str = st.text_input("BBox (minx,miny,maxx,maxy, WGS84)", "22.0,44.3,40.3,52.5")
    n_x = st.number_input("Тайлов по X", min_value=1, max_value=10, value=4)
    n_y = st.number_input("Тайлов по Y", min_value=1, max_value=10, value=4)
    simplify_tol = st.number_input("Simplify tolerance (м)", min_value=0.0, value=3.0)
    snap_threshold = st.number_input("Порог привязки станций (м)", min_value=1.0, value=100.0)

col_a, col_b = st.sidebar.columns(2)
build_clicked = col_a.button("Построить / загрузить граф")
rebuild_clicked = col_b.button("Пересчитать заново")

if build_clicked or rebuild_clicked:
    force = rebuild_clicked
    with st.spinner("Строим граф... для всей страны это может занять несколько минут."):
        try:
            bbox = tuple(float(v) for v in bbox_str.split(',')) if use_tiling else None
            G, station_nodes = gl.get_or_build_graph(
                railways_path, stations_path, cache_path,
                full_bbox=bbox, n_x=int(n_x), n_y=int(n_y),
                force_rebuild=force, snap_threshold=snap_threshold,
                simplify_tolerance=simplify_tol,
            )
            st.session_state.G_base = G
            st.session_state.station_nodes = station_nodes
            st.session_state.capacity_overrides = {}
            st.session_state.failed_nodes = set()
            st.session_state.failed_edges = set()
            st.sidebar.success(f"Готово: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер.")
        except Exception as e:
            st.sidebar.error(f"Ошибка построения графа: {e}")

if st.session_state.G_base is None:
    st.info("Слева задайте пути к файлам и нажмите «Построить / загрузить граф», чтобы начать.")
    st.stop()

G_base = st.session_state.G_base
station_nodes = st.session_state.station_nodes
station_names = sorted(station_nodes.keys())

# ---------------------------------------------------------------------------
# Сайдбар: выбор станций и сценарий
# ---------------------------------------------------------------------------
st.sidebar.header("2. Маршрут")
start_name = st.sidebar.selectbox("Станция-источник", station_names, index=0)
end_name = st.sidebar.selectbox("Станция-сток", station_names, index=min(1, len(station_names) - 1))

st.sidebar.header("3. Отказы (что вывести из строя)")
failed_station_names = st.sidebar.multiselect("Отключить станции", station_names)
st.session_state.failed_nodes = {station_nodes[n] for n in failed_station_names}

if st.sidebar.button("Сбросить отказы"):
    st.session_state.failed_nodes = set()
    st.session_state.failed_edges = set()

st.sidebar.caption(
    f"Отключено станций: {len(st.session_state.failed_nodes)} | "
    f"Отключено участков: {len(st.session_state.failed_edges)}"
)

# ---------------------------------------------------------------------------
# Основной расчёт: путь + max-flow, до/после отказа
# ---------------------------------------------------------------------------
st.title("Симулятор железнодорожной сети: нагрузка и отказы")

run = st.button("Рассчитать (путь + max-flow)", type="primary")

G_after = graph_with_failures()

if run:
    try:
        comparison = gl.compare_before_after(working_graph(), G_after, station_nodes, start_name, end_name)
        st.session_state.last_result = comparison
    except Exception as e:
        st.error(f"Ошибка расчёта: {e}")
        st.session_state.last_result = None

result = st.session_state.last_result

col1, col2 = st.columns(2)
if result:
    for col, label, title in ((col1, 'before', 'До отказа'), (col2, 'after', 'После отказа')):
        with col:
            st.subheader(title)
            r = result[label]
            if not r['connected']:
                st.error("Станции НЕ связаны (сеть разорвана).")
            else:
                st.metric("Длина кратчайшего пути", f"{r['path_length_km']:.1f} км")
                st.metric("Max-flow (пропускная способность)", f"{r['max_flow']:.1f}")

    if result['before']['connected'] and result['after']['connected']:
        delta_len = result['after']['path_length_km'] - result['before']['path_length_km']
        delta_flow = result['after']['max_flow'] - result['before']['max_flow']
        st.write(
            f"**Изменение при отказе:** длина маршрута {delta_len:+.1f} км, "
            f"max-flow {delta_flow:+.1f}."
        )

# ---------------------------------------------------------------------------
# Карта
# ---------------------------------------------------------------------------
st.subheader("Карта сети")

show_full_network = st.checkbox("Показывать все рёбра сети (может тормозить на большой сети)", value=False)

# min-cut для подсветки узкого места (считаем на графе С отказами, чтобы видеть
# актуальное узкое место в текущем сценарии)
cut_edges = set()
path_nodes = []
s, t = station_nodes.get(start_name), station_nodes.get(end_name)
if s in G_after and t in G_after and nx.has_path(G_after, s, t):
    path_nodes = nx.shortest_path(G_after, s, t, weight='weight')
    _, _, cut_edges_list, _ = gl.max_flow_between(G_after, s, t)
    cut_edges = {frozenset(e) for e in cut_edges_list}

center_lat, center_lon = xy_to_latlon(*list(G_base.nodes)[0])
m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="cartodbpositron")

edges_to_draw = G_after.edges(data=True) if show_full_network else []
if not show_full_network and path_nodes:
    # рисуем только маршрут + его окрестность (соседние рёбра узлов маршрута) — быстрее
    node_set = set(path_nodes)
    for n in path_nodes:
        node_set.update(G_after.neighbors(n))
    edges_to_draw = [(u, v, d) for u, v, d in G_after.edges(data=True) if u in node_set or v in node_set]

for u, v, data in edges_to_draw:
    is_cut = frozenset((u, v)) in cut_edges
    color = 'red' if is_cut else 'gray'
    weight = 4 if is_cut else 2
    lat1, lon1 = xy_to_latlon(*u)
    lat2, lon2 = xy_to_latlon(*v)
    folium.PolyLine(
        [(lat1, lon1), (lat2, lon2)], color=color, weight=weight, opacity=0.8,
        tooltip=f"capacity={data.get('capacity', '?')}, длина={data.get('weight', 0)/1000:.1f} км",
    ).add_to(m)

if path_nodes:
    coords = [xy_to_latlon(*n) for n in path_nodes]
    folium.PolyLine(coords, color='green', weight=5, opacity=0.9, tooltip="Кратчайший путь").add_to(m)

for name, coord in station_nodes.items():
    lat, lon = xy_to_latlon(*coord)
    is_failed = coord in st.session_state.failed_nodes
    color = 'black' if is_failed else ('lime' if name == start_name else ('purple' if name == end_name else 'blue'))
    folium.CircleMarker(
        [lat, lon], radius=5, color=color, fill=True, fill_opacity=0.9, tooltip=name,
    ).add_to(m)

map_data = st_folium(m, width=1100, height=650)

# ---------------------------------------------------------------------------
# Клик по карте → выбор ближайшего ребра для отключения / изменения capacity
# ---------------------------------------------------------------------------
st.subheader("Отключить участок / изменить пропускную способность по клику")
st.caption("Кликните на карте рядом с нужным участком пути, затем выберите действие ниже.")

if map_data and map_data.get('last_clicked'):
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    click_x, click_y = latlon_to_xy(lat, lon)

    best_edge, best_dist = None, float('inf')
    for u, v in G_after.edges():
        # расстояние точки до отрезка (грубая, но быстрая оценка через сегмент)
        import numpy as np
        p = np.array([click_x, click_y])
        a, b = np.array(u), np.array(v)
        ab = b - a
        t_param = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
        proj = a + t_param * ab
        d = np.linalg.norm(p - proj)
        if d < best_dist:
            best_dist, best_edge = d, (u, v)

    if best_edge and best_dist < 2000:  # в пределах 2 км от клика
        st.session_state.selected_edge = best_edge
        u, v = best_edge
        cap = G_after[u][v].get('capacity', 1.0)
        st.write(f"Выбран участок: длина {G_after[u][v]['weight']/1000:.2f} км, текущая capacity = {cap}")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Отключить этот участок (отказ)"):
                st.session_state.failed_edges.add(best_edge)
                st.rerun()
        with c2:
            new_cap = st.number_input("Новая пропускная способность", min_value=0.0, value=float(cap), key="new_cap_input")
        with c3:
            if st.button("Применить новую capacity"):
                st.session_state.capacity_overrides[best_edge] = new_cap
                st.rerun()
    else:
        st.write("Рядом с кликом не найдено участка пути (попробуйте кликнуть точнее на линию).")

# ---------------------------------------------------------------------------
# Статистика сети (справочно)
# ---------------------------------------------------------------------------
with st.expander("Статистика сети (текущий сценарий, с учётом отказов)"):
    stats = gl.network_statistics(G_after)
    st.write(f"Узлов: {stats['n_nodes']} | Рёбер: {stats['n_edges']} | "
             f"Компонент связности: {stats['n_components']} | "
             f"Крупнейшая компонента: {stats['component_sizes'][0] if stats['component_sizes'] else 0}")
    st.write("Топ узлов по betweenness centrality (узкие места сети):")
    st.dataframe([{"узел": str(n), "centrality": round(v, 4)} for n, v in stats['top_betweenness']])
