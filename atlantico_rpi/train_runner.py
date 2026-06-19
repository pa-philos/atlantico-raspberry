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
    parser.add_argument('--x-file', required=False)
    parser.add_argument('--y-file', required=False)
    parser.add_argument('--config-json', required=True)
    parser.add_argument('--output-metrics', required=False)
    parser.add_argument('--output-weights', required=False)
    parser.add_argument('--persistent', action='store_true', default=False)
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
        
        # Build Keras model initially to keep it cached in RAM
        util.build_keras_model()

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

    if args.persistent:
        print("READY", flush=True)
        # Loop reading commands from stdin as JSON lines
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd_data = json.loads(line)
            except Exception as e:
                print(f"ERROR invalid JSON: {str(e)}", flush=True)
                continue

            cmd = cmd_data.get('command')
            if cmd == 'exit':
                break
            elif cmd == 'train':
                x_file = cmd_data.get('x_file')
                y_file = cmd_data.get('y_file')
                weights_in_path = cmd_data.get('weights_in')
                weights_out_path = cmd_data.get('weights_out')
                metrics_out_path = cmd_data.get('metrics_out')

                if not x_file or not y_file or not weights_out_path or not metrics_out_path:
                    print("ERROR missing required fields in train command", flush=True)
                    continue

                try:
                    # Load weights if specified and exists
                    if weights_in_path and weights_in_path != "None" and os.path.exists(weights_in_path):
                        with open(weights_in_path, 'rb') as f:
                            weights_bytes = f.read()
                        
                        model_tf = getattr(util, '_last_trained_tf_model', None)
                        if model_tf is not None:
                            util.set_keras_model_weights(model_tf, weights_bytes)
                        else:
                            print("ERROR model not initialized", flush=True)
                            continue

                    # Run training
                    m = Model()
                    metrics = util.train_model_from_dataset(m, x_file, y_file)

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

                    with open(metrics_out_path, 'w') as f:
                        json.dump(metrics_dict, f)

                    # Serialize weights to bytes
                    tf_model = getattr(util, '_last_trained_tf_model', None)
                    if tf_model is not None:
                        raw_bytes = util.serialize_to_nn_bytes(keras_model=tf_model)
                        if raw_bytes:
                            with open(weights_out_path, 'wb') as f:
                                f.write(raw_bytes)

                    print("DONE", flush=True)

                except Exception as e:
                    traceback.print_exc()
                    print(f"ERROR {str(e)}", flush=True)
            else:
                print(f"ERROR unknown command {cmd}", flush=True)
    else:
        # One-shot fallback mode
        try:
            m = Model()
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
