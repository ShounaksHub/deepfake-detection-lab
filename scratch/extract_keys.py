import ast

with open(r"C:\Users\shoun\OneDrive\Desktop\hackathon\brainwave hackathon\deepfake_detectorV3.py", "r", encoding="utf-8") as f:
    source = f.read()

tree = ast.parse(source)

model_ids = []
thumb_keys = []

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id == "MODEL_REGISTRY":
                    if isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(value, ast.Dict):
                                for k, v in zip(value.keys, value.values):
                                    if isinstance(k, ast.Constant) and k.value == "model_id":
                                        if isinstance(v, ast.Constant):
                                            model_ids.append(v.value)
                elif target.id == "THUMBNAILS":
                    if isinstance(node.value, ast.Dict):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant):
                                thumb_keys.append(key.value)

print("MODELS:", model_ids)
print("THUMBS:", thumb_keys)
