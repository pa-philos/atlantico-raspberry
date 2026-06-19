#!/usr/bin/env python3
import sys
import os
import json
import argparse
import traceback

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from atlantico_rpi.model_util import ModelConfig, ModelUtil, Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-file', required=True)
    parser.add_argument('--y-file', required=True)
    parser.add_argument('--config-json', required=True)
    parser.add_argument('--output-metrics', required=True)
    parser.add_argument('--output-weights', required=True)
    args = parser.parse_args()

    try:
        # Load config dict from JSON
        cfg_dict = json.loads(args.config_json)
        mc = ModelConfig(
            layers=cfg_dict.get('layers', [10, 10]),
            activation_functions=cfg_dict.get('activation_functions', [0, 0]),
            epochs=cfg_dict.get('epochs', 1),
            random_seed=cfg_dict.get('random_seed', 10),
            learning_rate_of_weights=cfg_dict.get('learning_rate_of_weights', 0.3333),
            learning_rate_of_biases=cfg_dict.get('learning_rate_of_biases', 0.0666),
            json_weights=cfg_dict.get('json_weights', False)
        )

        util = ModelUtil(mc)
        m = Model()

        # Run training
        metrics = util.train_model_from_dataset(m, args.x_file, args.y_file)

        # Serialize metrics to JSON
        metrics_dict = {
            'numberOfClasses': getattr(metrics, 'numberOfClasses', 0),
            'meanSqrdError': getattr(metrics, 'meanSqrdError', 0.0),
            'parsingTime': getattr(metrics, 'parsingTime', 0.0),
            'trainingTime': getattr(metrics, 'trainingTime', 0.0),
            'epochs': getattr(metrics, 'epochs', 0),
            'datasetSize': getattr(metrics, 'datasetSize', 0),
            'accuracy': getattr(metrics, 'accuracy', 0.0),
            'precision': getattr(metrics, 'precision', 0.0),
            'recall': getattr(metrics, 'recall', 0.0),
            'f1Score': getattr(metrics, 'f1Score', 0.0),
            'balancedAccuracy': getattr(metrics, 'balancedAccuracy', 0.0),
            'balancedPrecision': getattr(metrics, 'balancedPrecision', 0.0),
            'balancedRecall': getattr(metrics, 'balancedRecall', 0.0),
            'balancedF1Score': getattr(metrics, 'balancedF1Score', 0.0),
        }

        # Per-class classifier metrics
        per_class = []
        if getattr(metrics, 'metrics', None):
            for cm in metrics.metrics:
                per_class.append({
                    'truePositives': cm.truePositives,
                    'trueNegatives': cm.trueNegatives,
                    'falsePositives': cm.falsePositives,
                    'falseNegatives': cm.falseNegatives
                })
        metrics_dict['metrics'] = per_class

        with open(args.output_metrics, 'w') as f:
            json.dump(metrics_dict, f)

        # Serialize weights to bytes
        tf_model = getattr(util, '_last_trained_tf_model', None)
        if tf_model is not None:
            raw_bytes = util.serialize_to_nn_bytes(keras_model=tf_model)
            if raw_bytes:
                with open(args.output_weights, 'wb') as f:
                    f.write(raw_bytes)

        sys.exit(0)

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
