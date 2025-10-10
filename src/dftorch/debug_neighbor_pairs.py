import torch

def find_duplicate_and_symmetric_pairs(neighbor_I, neighbor_J):
    """
    Print duplicate and symmetric (i,j)-(j,i) pairs in the neighbor list.
    """
    pairs = torch.stack([neighbor_I, neighbor_J], dim=1).cpu().numpy()
    pair_tuples = [tuple(row) for row in pairs]
    from collections import Counter
    counts = Counter(pair_tuples)
    print("Duplicate pairs (i,j) with count > 1:")
    for pair, count in counts.items():
        if count > 1:
            print(f"  {pair}: {count} times")
    # Find symmetric pairs
    pair_set = set(pair_tuples)
    symmetric = set()
    for i, j in pair_set:
        if (j, i) in pair_set and (j, i) != (i, j):
            symmetric.add(tuple(sorted((i, j))))
    print("Symmetric pairs (i,j)-(j,i):")
    for pair in symmetric:
        print(f"  {pair}")

# Example usage in your workflow:
# find_duplicate_and_symmetric_pairs(neighbor_I, neighbor_J)
