import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def merge_event_files_with_offset(input_files, output_dir):
    """
    Merge multiple TensorBoard event files, offsetting steps so they
    appear as one continuous run rather than multiple runs starting at 0.
    """
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    step_offset = 0
    max_step = 0

    for input_file in sorted(input_files):
        print(f"Processing {input_file}...")

        ea = EventAccumulator(input_file)
        ea.Reload()

        tags = ea.Tags().get('scalars', [])
        if not tags:
            print(f"  No scalar tags found, skipping.")
            continue

        # Find the max step in this file to compute the next offset
        file_max_step = 0
        for tag in tags:
            events = ea.Scalars(tag)
            if events:
                file_max_step = max(file_max_step, max(e.step for e in events))

        # Write all scalars with the offset applied
        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                writer.add_scalar(tag, e.value, e.step + step_offset)

        print(f"  Tags: {tags}")
        print(f"  Steps: 0 → {file_max_step}, offset by {step_offset}")

        # Next file starts where this one ended
        step_offset += file_max_step + 1

    writer.close()
    print(f"\nDone. Merged output written to {output_dir}")


# ── Config ────────────────────────────────────────────────────────────────────
input_files = sorted(glob.glob(
    "/home/emeralddawns/Documents/StreetFighter2RL/logs/exp1/to_be_merged/**/events.out.tfevents.*",
    recursive=True
))

print(f"Found {len(input_files)} event files:")
for f in input_files:
    print(f"  {f}")

merge_event_files_with_offset(
    input_files,
    output_dir="/home/emeralddawns/Documents/StreetFighter2RL/logs/exp1/merged"
)