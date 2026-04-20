import cv2
import numpy as np
import time
import logging
from contextlib import contextmanager
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from graphics import Rect, crop_image
from common import MarkDirection, detect_mark_direction, remove_small_features, get_ccs, strip_zeros_2d, entity
from npext import npext, to_gray, bin_threshold, dilate, erode, gaussian_blur


@dataclass
class EntityBBox:
    top_left: np.ndarray
    bottom_right: np.ndarray
    size: np.ndarray
    grid_cell: np.ndarray
    tags: set = field(default_factory=set)

    @property
    def rect(self) -> Rect:
        return Rect(self.top_left[0], self.top_left[1], 
                   self.bottom_right[0] - self.top_left[0],
                   self.bottom_right[1] - self.top_left[1])

    @property
    def center(self) -> np.ndarray:
        return (self.top_left + self.bottom_right) // 2


def deduce_frame_offset(frame1: np.ndarray,
                        frame2: np.ndarray,
                        roi: Optional[Rect] = None
                        ) -> Tuple[np.ndarray, float]:
    """Deduce pixel displacement between two consecutive Factorio frames.

    In Factorio the character stays fixed on screen while the world scrolls.
    Returns (dx, dy) such that frame2 content ≈ frame1 shifted by (dx, dy).
    A point at (x, y) in frame1 appears at approximately (x - dx, y - dy)
    in frame2.

    Uses gradient-phase correlation: Sobel gradients eliminate additive
    brightness changes (daytime cycle) and the Hann window reduces spectral
    leakage from image borders.

    Args:
        frame1: Previous frame (BGR, uint8)
        frame2: Current frame (BGR, uint8)
        roi: Optional region of interest (avoids UI elements)

    Returns:
        (offset, confidence) where offset = np.array([dx, dy]) and
        confidence is the phase-correlation peak response (higher = better).
    """
    f1 = crop_image(frame1, roi) if roi is not None else frame1
    f2 = crop_image(frame2, roi) if roi is not None else frame2

    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    gx1 = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
    grad1 = np.sqrt(gx1 * gx1 + gy1 * gy1)

    gx2 = cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3)
    grad2 = np.sqrt(gx2 * gx2 + gy2 * gy2)

    h, w = grad1.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    shift, response = cv2.phaseCorrelate(grad1 * win, grad2 * win)

    dx, dy = float(shift[0]), float(shift[1])
    if dx > w / 2:
        dx -= w
    elif dx < -w / 2:
        dx += w
    if dy > h / 2:
        dy -= h
    elif dy < -h / 2:
        dy += h

    return np.array([dx, dy]), float(response)


def filter_marker_mask(img: np.ndarray,
                       min_brightness: int = 50,
                       max_brightness: int = 255,
                       min_greenness: int = 30) -> np.ndarray:
    """Filter mask by pixel brightness and greenness.

    Args:
        img: BGR image (uint8, HxWx3)
        min_brightness: Minimum V (brightness) threshold
        max_brightness: Maximum V (brightness) threshold
        min_greenness: Minimum G - max(R,B) difference required

    Returns:
        Binary mask (uint8, HxW) where 255 = passes both filters
    """
    b, g, r = cv2.split(img)
    greenness = g.astype(np.int16) - np.maximum(r, b).astype(np.int16)
    green_mask = (greenness >= min_greenness).astype(np.uint8) * 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    brightness_mask = ((v >= min_brightness) & (v <= max_brightness)).astype(np.uint8) * 255

    return cv2.bitwise_and(green_mask, brightness_mask)


@dataclass
class GuessedGrid:
    ox: int
    oy: int
    cell_width: int
    vl: np.ndarray
    hl: np.ndarray
    score: float
    marker_positions: np.ndarray

    def cell_at(self, x: int, y: int) -> Tuple[int, int]:
        j = round((x - self.ox) / self.cell_width)
        i = round((y - self.oy) / self.cell_width)
        return (j, i)


def _grid_cost(ox: int, oy: int, cw: int,
               col_weights: np.ndarray, row_weights: np.ndarray) -> float:
    """Count weighted mask pixels sitting on grid lines for a given origin."""
    w = len(col_weights)
    h = len(row_weights)
    vcols = np.arange(ox % cw, w, cw)
    hrows = np.arange(oy % cw, h, cw)
    return float(col_weights[vcols].sum() + row_weights[hrows].sum())


def _make_grid(ox: int, oy: int, cw: int,
               mask: np.ndarray, cost: float,
               positions: np.ndarray) -> GuessedGrid:
    h, w = mask.shape[:2]
    x0 = max(0, int(positions[:, 0].min()) - cw)
    y0 = max(0, int(positions[:, 1].min()) - cw)
    x1 = min(w, int(positions[:, 0].max()) + cw)
    y1 = min(h, int(positions[:, 1].max()) + cw)
    vl = np.arange(ox + (x0 - ox) // cw * cw, x1 + 1, cw)
    hl = np.arange(oy + (y0 - oy) // cw * cw, y1 + 1, cw)
    return GuessedGrid(ox=ox, oy=oy, cell_width=cw,
                       vl=vl, hl=hl, score=cost,
                       marker_positions=positions)


def guess_grid_bruteforce(mask: np.ndarray,
                           cell_widths: Tuple[int, ...] = (32, 64)
                           ) -> Optional[GuessedGrid]:
    """Bruteforce search for grid (ox, oy, cell_width) minimizing mask intersections.

    Grid lines at x = ox + k*cell_width and y = oy + k*cell_width are laid over
    the mask. The origin and cell_width that produce the fewest mask-pixel
    intersections means the grid lines fall between markers instead of through them.

    Cost is decomposed into independent x and y components:
      col_weight[x] = sum of mask pixels in column x
      row_weight[y] = sum of mask pixels in row y
    so trying all cw*2 offsets is O(cw * (W + H) / cw).

    Args:
        mask: Binary mask (uint8, HxW) from filter_marker_mask
        cell_widths: Candidate grid cell widths

    Returns:
        GuessedGrid or None if mask is empty
    """
    binary = (mask > 0).astype(np.float64)
    if binary.sum() == 0:
        return None
    h, w = mask.shape[:2]

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num_labels <= 1:
        return None
    positions = centroids[1:].astype(int)

    col_w = binary.sum(axis=0)
    row_w = binary.sum(axis=1)

    best = None
    best_cost = float('inf')

    for cw in cell_widths:
        for ox in range(cw):
            vcols = np.arange(ox, w, cw)
            if len(vcols) == 0:
                continue
            cx = float(col_w[vcols].sum())
            for oy in range(cw):
                hrows = np.arange(oy, h, cw)
                if len(hrows) == 0:
                    continue
                cy = float(row_w[hrows].sum())
                cost = cx + cy
                if cost < best_cost:
                    best_cost = cost
                    best = _make_grid(ox, oy, cw, mask, cost, positions)

    if best is not None:
        best.score = best_cost
    return best


def guess_grid_ga(mask: np.ndarray,
                   cell_widths: Tuple[int, ...] = (32, 64),
                   population_size: int = 30,
                   generations: int = 60,
                   mutation_range: int = 5,
                   elite_fraction: float = 0.2,
                   seed: Optional[int] = None) -> Optional[GuessedGrid]:
    """Genetic algorithm to find grid origin minimizing mask intersections.

    Population of (ox, oy, cw) individuals evolves via selection, crossover
    and mutation. Fitness = negative intersection cost (lower is better).
    See guess_grid_bruteforce for cost definition.

    Args:
        mask: Binary mask (uint8, HxW) from filter_marker_mask
        cell_widths: Candidate grid cell widths
        population_size: Number of individuals per generation
        generations: Number of evolution iterations
        mutation_range: Max pixel offset a mutation can shift
        elite_fraction: Fraction of top individuals kept each generation
        seed: Random seed for reproducibility

    Returns:
        GuessedGrid or None if mask is empty
    """
    rng = np.random.default_rng(seed)

    binary = (mask > 0).astype(np.float64)
    if binary.sum() == 0:
        return None
    h, w = mask.shape[:2]

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if num_labels <= 1:
        return None
    positions = centroids[1:].astype(int)

    col_w = binary.sum(axis=0)
    row_w = binary.sum(axis=1)

    def cost(ox: int, oy: int, cw: int) -> float:
        vcols = np.arange(ox, w, cw)
        hrows = np.arange(oy, h, cw)
        return float(col_w[vcols].sum() + row_w[hrows].sum())

    cw_list = list(cell_widths)

    pop_ox = np.array([rng.integers(0, cw) for cw in
                        [cw_list[rng.integers(0, len(cw_list))] for _ in range(population_size)]])
    pop_oy = np.array([rng.integers(0, cw) for cw in
                        [cw_list[rng.integers(0, len(cw_list))] for _ in range(population_size)]])
    pop_cw = np.array([cw_list[rng.integers(0, len(cw_list))] for _ in range(population_size)])

    n_elite = max(1, int(population_size * elite_fraction))

    for _ in range(generations):
        costs = np.array([cost(pop_ox[i], pop_oy[i], pop_cw[i])
                          for i in range(population_size)])
        order = np.argsort(costs)
        costs = costs[order]
        pop_ox = pop_ox[order]
        pop_oy = pop_oy[order]
        pop_cw = pop_cw[order]

        new_ox = list(pop_ox[:n_elite])
        new_oy = list(pop_oy[:n_elite])
        new_cw = list(pop_cw[:n_elite])

        while len(new_ox) < population_size:
            p1, p2 = rng.integers(0, n_elite, size=2)
            cw_child = pop_cw[p1]
            ox_child = int(rng.integers(0, cw_child))
            oy_child = int(rng.integers(0, cw_child))
            if rng.random() < 0.5:
                ox_child = int(pop_ox[p1]) + int(rng.integers(-mutation_range, mutation_range + 1))
            else:
                ox_child = int(pop_ox[p2]) + int(rng.integers(-mutation_range, mutation_range + 1))
            if rng.random() < 0.5:
                oy_child = int(pop_oy[p1]) + int(rng.integers(-mutation_range, mutation_range + 1))
            else:
                oy_child = int(pop_oy[p2]) + int(rng.integers(-mutation_range, mutation_range + 1))
            ox_child = ox_child % cw_child
            oy_child = oy_child % cw_child
            if rng.random() < 0.1:
                cw_child = cw_list[rng.integers(0, len(cw_list))]
                ox_child = int(rng.integers(0, cw_child))
                oy_child = int(rng.integers(0, cw_child))
            new_ox.append(ox_child)
            new_oy.append(oy_child)
            new_cw.append(cw_child)

        pop_ox = np.array(new_ox)
        pop_oy = np.array(new_oy)
        pop_cw = np.array(new_cw)

    costs = np.array([cost(pop_ox[i], pop_oy[i], pop_cw[i])
                      for i in range(population_size)])
    best_idx = int(np.argmin(costs))
    best_ox = int(pop_ox[best_idx])
    best_oy = int(pop_oy[best_idx])
    best_cw = int(pop_cw[best_idx])
    best_cost = float(costs[best_idx])

    return _make_grid(best_ox, best_oy, best_cw, mask, best_cost, positions)


def clean_green_mask(mask: np.ndarray, 
                     min_area: int = 50,
                     erode_size: int = 2,
                     dilate_size: int = 3) -> np.ndarray:
    """Clean up green mask by removing noise and filling gaps.
    
    Args:
        mask: Binary mask
        min_area: Minimum connected component area to keep
        erode_size: Size of erosion kernel
        dilate_size: Size of dilation kernel
        
    Returns:
        Cleaned binary mask
    """
    cleaned = remove_small_features(mask, min_area)
    cleaned = cv2.erode(cleaned, cv2.getStructuringElement(cv2.MORPH_RECT, (2*erode_size+1, 2*erode_size+1)))
    cleaned = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilate_size+1, 2*dilate_size+1)))
    return cleaned


def detect_corner_markers(mask: np.ndarray, 
                          marker_size: int = 14) -> List[Tuple[np.ndarray, MarkDirection]]:
    """Detect corner markers in the cleaned green mask.
    
    Corner markers appear as small filled squares at entity corners in blueprint overlay.
    
    Args:
        mask: Cleaned binary mask
        marker_size: Expected size of corner markers
        
    Returns:
        List of (position, direction) tuples for detected markers
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    markers = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4 or area > marker_size * marker_size:
            continue
            
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        if abs(w - h) > 3:
            continue
            
        center = np.array([x + w // 2, y + h // 2])
        markers.append((center, area))
    
    return markers


def cluster_markers_to_corners(markers: List[Tuple[np.ndarray, int]], 
                               cluster_threshold: int = 20) -> List[Tuple[np.ndarray, MarkDirection]]:
    """Cluster nearby markers and determine corner directions.
    
    Args:
        markers: List of (position, area) tuples
        cluster_threshold: Maximum distance to consider markers as same corner
        
    Returns:
        List of (corner_position, MarkDirection) tuples
    """
    if not markers:
        return []
    
    visited = [False] * len(markers)
    corners = []
    
    for i, (pos_i, area_i) in enumerate(markers):
        if visited[i]:
            continue
        
        cluster = [(pos_i, area_i)]
        visited[i] = True
        
        for j, (pos_j, area_j) in enumerate(markers):
            if not visited[j] and np.linalg.norm(pos_i - pos_j) < cluster_threshold:
                cluster.append((pos_j, area_j))
                visited[j] = True
        
        avg_pos = np.mean([p for p, _ in cluster], axis=0).astype(int)
        corners.append(avg_pos)
    
    if len(corners) < 2:
        return []
    
    corners = np.array(corners)
    min_y_idx = np.argmin(corners[:, 1])
    max_y_idx = np.argmax(corners[:, 1])
    min_x_idx = np.argmin(corners[:, 0])
    max_x_idx = np.argmax(corners[:, 0])
    
    result = []
    for idx, pos in enumerate(corners):
        is_top = idx == min_y_idx or abs(pos[1] - corners[min_y_idx][1]) < 10
        is_bottom = idx == max_y_idx or abs(pos[1] - corners[max_y_idx][1]) < 10
        is_left = idx == min_x_idx or abs(pos[0] - corners[min_x_idx][0]) < 10
        is_right = idx == max_x_idx or abs(pos[0] - corners[max_x_idx][0]) < 10
        
        if is_top and is_left:
            result.append((pos, MarkDirection.TOP_LEFT))
        elif is_top and is_right:
            result.append((pos, MarkDirection.TOP_RIGHT))
        elif is_bottom and is_left:
            result.append((pos, MarkDirection.BOTTOM_LEFT))
        elif is_bottom and is_right:
            result.append((pos, MarkDirection.BOTTOM_RIGHT))
    
    return result


def deduce_entity_bboxes_from_corners(corners: List[Tuple[np.ndarray, MarkDirection]],
                                      grid_vl: Optional[np.ndarray] = None,
                                      grid_hl: Optional[np.ndarray] = None,
                                      cell_width: int = 32) -> List[EntityBBox]:
    """Deduce entity bounding boxes from corner markers.
    
    Args:
        corners: List of (position, MarkDirection) tuples
        grid_vl: Vertical grid lines positions (optional)
        grid_hl: Horizontal grid lines positions (optional)
        cell_width: Width of a grid cell in pixels
        
    Returns:
        List of EntityBBox objects
    """
    if len(corners) < 2:
        return []
    
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    
    for pos, direction in corners:
        if direction == MarkDirection.TOP_LEFT:
            top_left = pos
        elif direction == MarkDirection.TOP_RIGHT:
            top_right = pos
        elif direction == MarkDirection.BOTTOM_LEFT:
            bottom_left = pos
        elif direction == MarkDirection.BOTTOM_RIGHT:
            bottom_right = pos
    
    if top_left is None or bottom_right is None:
        return []
    
    if top_right is None:
        top_right = np.array([bottom_right[0], top_left[1]])
    if bottom_left is None:
        bottom_left = np.array([top_left[0], bottom_right[1]])
    
    entity = EntityBBox(
        top_left=top_left,
        bottom_right=bottom_right,
        size=np.array([
            max(1, (bottom_right[0] - top_left[0] + cell_width // 2) // cell_width),
            max(1, (bottom_right[1] - top_left[1] + cell_width // 2) // cell_width)
        ]),
        grid_cell=np.array([
            top_left[0] // cell_width,
            top_left[1] // cell_width
        ])
    )
    
    return [entity]


def get_entity_bboxes_from_mask(mask: np.ndarray,
                                grid_vl: Optional[np.ndarray] = None,
                                grid_hl: Optional[np.ndarray] = None,
                                cell_width: int = 32,
                                min_entity_area: int = 100) -> List[EntityBBox]:
    """Extract entity bounding boxes from a cleaned green mask.
    
    This function combines marker detection and bbox deduction.
    
    Args:
        mask: Cleaned binary mask with green overlay
        grid_vl: Vertical grid lines positions (optional)
        grid_hl: Horizontal grid lines positions (optional)
        cell_width: Width of a grid cell in pixels
        min_entity_area: Minimum area for an entity
        
    Returns:
        List of EntityBBox objects
    """
    markers = detect_corner_markers(mask)
    corners = cluster_markers_to_corners(markers)
    entities = deduce_entity_bboxes_from_corners(corners, grid_vl, grid_hl, cell_width)
    return entities


@contextmanager
def blueprint_selection(snail, 
                       selection_rect: Rect,
                       sleep_before: float = 0.1,
                       sleep_after: float = 0.2,
                       mouse_spd: int = 0):
    """Context manager for blueprint selection area.
    
    Opens blueprint tool, performs drag selection, captures frame, then cancels.
    
    Args:
        snail: Snail instance
        selection_rect: Rectangle to select in screen coordinates
        sleep_before: Time to wait before starting selection
        sleep_after: Time to wait after selection for rendering
        mouse_spd: Mouse movement speed (0 for instant)
        
    Yields:
        Captured frame as numpy array (BGR)
    """
    time.sleep(sleep_before)
    
    xy = np.array(selection_rect.xy())
    wh = np.array(selection_rect.wh())
    
    snail.ahk.send_input('b')
    time.sleep(sleep_after)
    
    snail.ahk.mouse_move(*xy, speed=mouse_spd)
    snail.ahk.click(button='L', direction='D')
    time.sleep(0.05)
    snail.ahk.mouse_move(*(xy + wh), speed=mouse_spd)
    snail.ahk.click(button='L', direction='U')
    time.sleep(sleep_after)
    
    frame = snail.wait_next_frame()
    
    yield frame
    
    snail.ahk.send_input('{Esc}')
    time.sleep(sleep_after)


def detect_entities_in_rect(snail,
                           selection_rect: Rect,
                           grid_vl: Optional[np.ndarray] = None,
                           grid_hl: Optional[np.ndarray] = None,
                           cell_width: int = 32,
                           use_hsv: bool = True,
                           debug: bool = False) -> Tuple[np.ndarray, List[EntityBBox]]:
    """Detect entities in a screen region using blueprint selection.
    
    Main entry point for entity detection feature.
    
    Args:
        snail: Snail instance
        selection_rect: Region to scan in screen coordinates
        grid_vl: Vertical grid lines (optional, for grid alignment)
        grid_hl: Horizontal grid lines (optional, for grid alignment)
        cell_width: Grid cell width in pixels
        use_hsv: Use HSV color space for green extraction (more robust)
        debug: If True, return debug image with detected entities drawn
        
    Returns:
        Tuple of (frame, list of EntityBBox)
    """
    with blueprint_selection(snail, selection_rect) as frame:
        if use_hsv:
            green_mask = extract_green_mask_hsv(frame)
        else:
            green_mask = extract_green_mask(frame)
        
        cleaned_mask = clean_green_mask(green_mask)
        
        entities = get_entity_bboxes_from_mask(
            cleaned_mask, 
            grid_vl, grid_hl, 
            cell_width
        )
        
        if debug and entities:
            debug_frame = frame.copy()
            for ent in entities:
                r = ent.rect
                cv2.rectangle(debug_frame, r.xy(), r.xy() + r.wh(), (0, 255, 0), 2)
                cv2.putText(debug_frame, f'{ent.size[0]}x{ent.size[1]}', 
                           (r.x0, r.y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return debug_frame, entities
        
        return frame, entities
