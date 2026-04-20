import h5py, json

# Check h5 structure
with h5py.File('data/iu_xray_artifacts/node_features_gpt2.h5', 'r') as f:
    def print_structure(name, obj):
        print(f"  {name}: {obj.shape if hasattr(obj, 'shape') else 'group'}")
    f.visititems(print_structure)
    # Check a sample
    print("\nTop-level keys:", list(f.keys()))

# Check node mapping
nm = json.load(open('data/iu_xray_artifacts/node_mapping.json'))
print(f"\nnode_mapping: {len(nm)} nodes")
print(f"Sample entries: {list(nm.items())[:5]}")