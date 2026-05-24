import os
import sys
import argparse
import torch

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
	sys.path.insert(0, path)

from data.dataload import load_df, split_X_y, split_train_test
from metrics.mse import rmse
from metrics.r2_score import r2_score
from models.mlp import MLP, MLPConfig
from models.decision_tree import DecisionTree, DecisionTreeConfig


def infer_mlp_config_from_state(state_dict: dict) -> MLPConfig:
	linear_layers = []
	for k, v in state_dict.items():
		if k.endswith(".weight") and k.startswith("network.") and getattr(v, "ndim", None) == 2:
			idx = int(k.split(".")[1])
			linear_layers.append((idx, v.shape))

	linear_layers.sort()
	if not linear_layers:
		raise ValueError("no linear layers found in state_dict")

	input_dim = linear_layers[0][1][1]
	hidden_dims = [shape[0] for _, shape in linear_layers[:-1]]
	output_dim = linear_layers[-1][1][0]

	return MLPConfig(model="mlp", input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)


def load_model_from_file(path: str):
	data = torch.load(path, map_location="cpu")

	# Decision tree was saved as {'tree': sklearn_tree}
	if isinstance(data, dict) and "tree" in data and not any(isinstance(v, torch.Tensor) for v in data.values()):
		cfg = DecisionTreeConfig(model="dt")
		model = DecisionTree(cfg)
		model.tree = data["tree"]
		return model, "dt"

	# MLP saved as state_dict (mapping of tensors)
	if isinstance(data, dict):
		if any(isinstance(v, torch.Tensor) for v in data.values()):
			cfg = infer_mlp_config_from_state(data)
			model = MLP(cfg)
			model.load_state_dict(data)
			return model, "mlp"

	raise ValueError(f"Unrecognized model file format: {path}")


def evaluate(args):
	data_path = os.path.join("data/processed", "usa_housing_dataset_processed.csv")
	df = load_df(data_path)
	X, y = split_X_y(df, args.target)

	(X_train, y_train), (X_test, y_test) = split_train_test(X, y)

	models_dir = args.models_dir
	files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pth")])
	if args.model:
		files = [f for f in files if args.model.lower() in f.lower()]

	if not files:
		print(f"No model files found in {models_dir}")
		return

	for fname in files:
		fpath = os.path.join(models_dir, fname)
		try:
			model, kind = load_model_from_file(fpath)
		except Exception as e:
			print(f"Skipping {fname}: failed to load ({e})")
			continue

		try:
			y_pred = model.predict(X_test)
		except Exception as e:
			print(f"Skipping {fname}: predict() failed ({e})")
			continue

		rmse_val = rmse(y_test, y_pred)
		r2_val = r2_score(y_test, y_pred)

		print(f"Model: {fname}  Type: {kind} \n  Test RMSE: {rmse_val:.4f}  Test R2: {r2_val:.4f}\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--models-dir", type=str, default=os.path.join("models", "best_models"))
	parser.add_argument("--model", type=str, default=None, help="optional substring to filter model filenames")
	parser.add_argument("--target", type=str, default="price")
	args = parser.parse_args()
	evaluate(args)

    