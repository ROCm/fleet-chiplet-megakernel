"""
Post-process task_graph.json to split large LINEAR tasks into multiple sub-tasks
for N-tile parallelization on AMD GPUs.
"""

import json
import copy
from pathlib import Path

# LINEAR task types that can be split
LINEAR_TASK_TYPES = {
    120,  # TASK_LINEAR
    108,  # TASK_LINEAR_WITH_RESIDUAL
}

# Configuration for different LINEAR variants
# variant_id -> (batch_size, output_size, reduction_size, output_stride)
# These need to match the variants registered in task_register.cc

def get_linear_config_from_variant(task_graph_path):
    """
    Extract LINEAR variant configurations by analyzing the task graph.
    Returns a dict mapping variant_id to output_size.
    """
    # For now, we'll use a heuristic based on common Qwen3-8B dimensions
    # In a real implementation, this would be extracted from the code generator
    return {
        0: 64,      # Small linear (QKV chunks)
        1: 256,     # Medium linear (gate_up chunks)
        2: 640,     # Large linear (lm_head chunks)
    }


def split_linear_task(task, n_tiles_per_worker=48, max_workers=8):
    """
    Split a LINEAR task into multiple sub-tasks.

    Args:
        task: The original LINEAR task dict
        n_tiles_per_worker: Number of N-tiles per worker (default 48)
        max_workers: Maximum number of workers to use (default 8)

    Returns:
        List of task dicts (split tasks or original if no split needed)
    """
    # Get output_size based on variant_id
    # These are the N dimensions from the kernel templates
    variant_configs = {
        # variant_id: (output_size, n_tile_size)
        0: (64, 64),       # OUTPUT_SIZE=64, processed in one tile
        1: (256, 64),      # OUTPUT_SIZE=256, 4 tiles
        2: (640, 64),      # OUTPUT_SIZE=640, 10 tiles
    }

    variant_id = task.get('variant_id', 0)
    if variant_id not in variant_configs:
        return [task]  # Unknown variant, don't split

    output_size, n_tile_size = variant_configs[variant_id]
    total_n_tiles = (output_size + n_tile_size - 1) // n_tile_size

    # Don't split if only a few tiles
    if total_n_tiles <= n_tiles_per_worker:
        # Set n_tile_start=0, n_tile_count=0 (meaning all tiles)
        task['n_tile_start'] = 0
        task['n_tile_count'] = 0
        return [task]

    # Calculate number of workers needed
    num_workers = min(max_workers, (total_n_tiles + n_tiles_per_worker - 1) // n_tiles_per_worker)
    tiles_per_worker = (total_n_tiles + num_workers - 1) // num_workers

    # Create sub-tasks
    sub_tasks = []
    for i in range(num_workers):
        n_tile_start = i * tiles_per_worker
        n_tile_count = min(tiles_per_worker, total_n_tiles - n_tile_start)

        if n_tile_count <= 0:
            break

        sub_task = copy.deepcopy(task)
        sub_task['n_tile_start'] = n_tile_start
        sub_task['n_tile_count'] = n_tile_count
        sub_tasks.append(sub_task)

    return sub_tasks


def split_task_graph(input_path, output_path=None, n_tiles_per_worker=48, max_workers=8):
    """
    Process a task_graph.json file and split large LINEAR tasks.

    Args:
        input_path: Path to input task_graph.json
        output_path: Path to output (default: overwrite input)
        n_tiles_per_worker: N-tiles per worker
        max_workers: Max workers per LINEAR task
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    with open(input_path, 'r') as f:
        task_graph = json.load(f)

    new_tasks = []
    split_count = 0

    for task in task_graph['all_tasks']:
        task_type = task.get('task_type', 0)

        if task_type in LINEAR_TASK_TYPES:
            sub_tasks = split_linear_task(task, n_tiles_per_worker, max_workers)
            if len(sub_tasks) > 1:
                split_count += 1
                print(f"Split LINEAR task (variant={task.get('variant_id')}) into {len(sub_tasks)} sub-tasks")
            new_tasks.extend(sub_tasks)
        else:
            # Ensure n_tile fields exist
            if 'n_tile_start' not in task:
                task['n_tile_start'] = 0
            if 'n_tile_count' not in task:
                task['n_tile_count'] = 0
            new_tasks.append(task)

    task_graph['all_tasks'] = new_tasks

    # Update event task ranges (this is simplified - may need more work)
    # For now, just note that events may need to be updated
    if split_count > 0:
        print(f"Warning: Split {split_count} LINEAR tasks. Event ranges may need adjustment.")

    with open(output_path, 'w') as f:
        json.dump(task_graph, f, indent=2)

    print(f"Wrote {len(new_tasks)} tasks to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split LINEAR tasks for N-tile parallelization')
    parser.add_argument('input', help='Input task_graph.json path')
    parser.add_argument('-o', '--output', help='Output path (default: overwrite input)')
    parser.add_argument('--tiles-per-worker', type=int, default=48,
                       help='N-tiles per worker (default: 48)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Max workers per LINEAR task (default: 8)')
    args = parser.parse_args()

    split_task_graph(args.input, args.output, args.tiles_per_worker, args.max_workers)
