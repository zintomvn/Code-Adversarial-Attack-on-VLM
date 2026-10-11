"""Fixed-area hotspot ranking with the original IoU fallback."""

import numpy as np


def compute_box_iou(box1, box2):
    # box: (x_min, x_max, y_min, y_max) inclusive
    x1_min, x1_max, y1_min, y1_max = box1
    x2_min, x2_max, y2_min, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min + 1)
    inter_h = max(0, inter_y_max - inter_y_min + 1)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    area2 = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def get_top_k_boxes_fixed_area(attention_map, target_ratio, k=1, iou_threshold=0.3, area_tolerance=0.05):
    """
    Find top-k boxes with maximum attention score that have area approximately equal to target_ratio * total_area.
    """
    H, W = attention_map.shape
    total_cells = H * W
    target_cells = int(target_ratio * total_cells)
    if target_cells < 1:
        target_cells = 1

    # When target ratio is effectively the full map, force search to stay near full area
    near_full = target_ratio >= 0.999
    area_lower = total_cells if near_full else max(1, int(target_cells * (1 - area_tolerance)))
    area_upper = total_cells if near_full else min(total_cells, int(target_cells * (1 + area_tolerance)))

    integral = np.cumsum(np.cumsum(attention_map, axis=0), axis=1)

    def get_sum(r1, c1, r2, c2):
        res = integral[r2, c2]
        if r1 > 0:
            res -= integral[r1 - 1, c2]
        if c1 > 0:
            res -= integral[r2, c1 - 1]
        if r1 > 0 and c1 > 0:
            res += integral[r1 - 1, c1 - 1]
        return res

    candidates = []

    for h in range(1, H + 1):
        w = int(round(target_cells / h))
        if w < 1:
            w = 1
        if w > W:
            w = W

        # Aspect ratio constraint to prevent elongated boxes
        if h / w > 3.0 or w / h > 3.0:
            continue

        area = h * w
        # Enforce area closeness to target to avoid undersized boxes dominating
        if area < area_lower or area > area_upper:
            continue

        for r in range(H - h + 1):
            for c in range(W - w + 1):
                current_sum = get_sum(r, c, r + h - 1, c + w - 1)
                # current_mean = current_sum / area
                candidates.append((current_sum, (c, c + w - 1, r, r + h - 1)))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Select top-k with IoU filtering
    selected_boxes = []
    for _, box in candidates:
        if len(selected_boxes) >= k:
            break

        # Check IoU with already selected boxes
        is_valid = True
        for selected_box in selected_boxes:
            if compute_box_iou(box, selected_box) > iou_threshold:
                is_valid = False
                break

        if is_valid:
            selected_boxes.append(box)

    # Fallback: if we could not gather k boxes under the IoU cap,
    # fill the rest with the next best candidates even if they overlap.
    if len(selected_boxes) < k and candidates:
        for _, box in candidates:
            if len(selected_boxes) >= k:
                break
            # Avoid exact duplicates
            if any(box == chosen for chosen in selected_boxes):
                continue
            selected_boxes.append(box)

    return selected_boxes


def build_schedule(attention_map, num_phases=10, top_k=3, iou_threshold=0.3):
    attention_map = np.asarray(attention_map)
    if attention_map.ndim != 2 or not attention_map.size or not np.isfinite(attention_map).all():
        raise ValueError("Attention must be a nonempty finite 2D map")
    if num_phases < 1 or top_k < 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid hotspot schedule parameters")
    grid_h, grid_w = attention_map.shape
    schedule = []
    for i in range(num_phases):
        ratio = (i + 1) / num_phases
        boxes = get_top_k_boxes_fixed_area(attention_map, ratio, k=top_k, iou_threshold=iou_threshold)
        for j, (xmin, xmax, ymin, ymax) in enumerate(boxes):
            top = np.clip(ymin / grid_h, 0.0, 1.0)
            left = np.clip(xmin / grid_w, 0.0, 1.0)
            height = np.clip((ymax - ymin + 1) / grid_h, 0.0, 1.0 - top)
            width = np.clip((xmax - xmin + 1) / grid_w, 0.0, 1.0 - left)
            schedule.append(
                dict(
                    threshold=ratio,
                    box=tuple(map(float, (top, left, height, width))),
                    phase_group=i + 1,
                    sub_phase=j + 1,
                )
            )
    if not schedule:
        raise ValueError("No hotspot candidates found")
    return schedule
