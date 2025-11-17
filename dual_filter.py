import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Process margin logs and select indices.")
    parser.add_argument("--file_before", type=str, required=True, help="Path to the 'before' margins file")
    parser.add_argument("--file_after", type=str, required=True, help="Path to the 'after' margins file")
    parser.add_argument("--output", type=str,
                        help="Output file for indices with both margins in range")
    args = parser.parse_args()

    # Load margins
    margins_before = np.loadtxt(args.file_before)
    margins_after = np.loadtxt(args.file_after)
    
    min_len = min(len(margins_before), len(margins_after))
    margins_before = margins_before[:min_len]
    margins_after = margins_after[:min_len]

    # Compute delta
    delta = margins_after - margins_before

    # Identify decreases and negatives
    indices_decrease = np.where(delta < 0)[0]
    indices_decrease_and_negative = indices_decrease[margins_before[indices_decrease] < 0]
    sorted_indices = np.sort(indices_decrease_and_negative).tolist() if len(indices_decrease_and_negative) > 0 else []

    # Range filter
    range_min = -10
    range_max = 10
    in_range_before = np.where((margins_before >= range_min) & (margins_before <= range_max))[0]
    in_range_after = np.where((margins_after >= range_min) & (margins_after <= range_max))[0]
    indices_both_in_range = np.intersect1d(in_range_before, in_range_after).tolist()

    # Combine and save final
    final_indices = sorted(set(sorted_indices) | set(indices_both_in_range))
    with open(args.output, "w") as f:
        for idx in final_indices:
            f.write(f"{idx}\n")
    print(f"Saved final indices to {args.output} ({len(final_indices)} entries)")

if __name__ == "__main__":
    main()
