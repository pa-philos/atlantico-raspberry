"""Model utilities and small persistence helpers."""

from dataclasses import dataclass
from typing import List, Optional, Any
import time
import json
import os
import numpy as np
import struct
from .config import X_TRAIN_PATH, Y_TRAIN_PATH
import logging

tf = None

def _lazy_import_tf():
    global tf
    if tf is None:
        try:
            import tensorflow as _tf
            tf = _tf
        except Exception:
            pass

_LOG = logging.getLogger(__name__)


@dataclass
class Model:
    biases: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    parsing_time: int = 0
    round: int = -1


@dataclass
class ClassClassifierMetrics:
    truePositives: int = 0
    trueNegatives: int = 0
    falsePositives: int = 0
    falseNegatives: int = 0


@dataclass
class MultiClassClassifierMetrics:
    metrics: Optional[List[ClassClassifierMetrics]] = None
    numberOfClasses: int = 0
    meanSqrdError: float = 0.0
    parsingTime: float = 0.0
    trainingTime: float = 0.0
    epochs: int = 0
    datasetSize: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1Score: float = 0.0
    balancedAccuracy: float = 0.0
    balancedPrecision: float = 0.0
    balancedRecall: float = 0.0
    balancedF1Score: float = 0.0


class ModelConfig:
    def __init__(self, layers: List[int], activation_functions: List[int], epochs: int = 1, random_seed: int = 10,
                 learning_rate_of_weights: float = 0.3333, learning_rate_of_biases: float = 0.0666,
                 json_weights: bool = False):
        self.layers = layers
        self.number_of_layers = len(layers)
        self.activation_functions = activation_functions
        self.epochs = epochs
        self.learning_rate_of_weights = learning_rate_of_weights
        self.learning_rate_of_biases = learning_rate_of_biases
        self.random_seed = random_seed
        self.json_weights = json_weights


class ModelUtil:
    """Utilities for model persistence and training stubs."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._last_trained_tf_model = None

    def save_model_to_disk(self, model: Model, file_path: str) -> bool:
        payload = {
            "biases": model.biases if model.biases is not None else [],
            "weights": model.weights if model.weights is not None else [],
            "parsingTime": int(model.parsing_time),
            "round": int(model.round),
        }
        dirpath = os.path.dirname(file_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return True

    def load_model_from_disk(self, file_path: str) -> Model:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = Model()
        m.biases = data.get("biases", [])
        m.weights = data.get("weights", [])
        m.parsing_time = int(data.get("parsingTime", 0))
        m.round = int(data.get("round", -1))
        return m

    def transform_data_to_model(self, stream) -> Model:
        """Transform an incoming stream (JSON or bytes) into a Model."""
        if stream is None:
            raise ValueError("stream must not be None")

        if isinstance(stream, dict):
            m = Model()
            m.biases = stream.get("biases", [])
            m.weights = stream.get("weights", [])
            m.parsing_time = int(stream.get("parsingTime", 0))
            m.round = int(stream.get("round", -1))
            return m

        if isinstance(stream, (bytes, bytearray)):
            parsed = self._parse_nn_bytes(bytes(stream))
            if parsed is None:
                m = Model()
                m.weights = [int(b) for b in stream]
                m.parsing_time = 0
                return m
            m = Model()
            biases = []
            weights = []
            for layer in parsed.get('layers', []):
                b = layer.get('biases', [])
                w = layer.get('weights', [])
                biases.extend(b.tolist() if isinstance(b, np.ndarray) else list(b))
                if isinstance(w, np.ndarray):
                    for row in w:
                        weights.extend(row.tolist())
                else:
                    weights.extend(w)
            m.biases = biases
            m.weights = weights
            m.parsing_time = int(parsed.get('parsingTime', 0))
            m.round = int(parsed.get('round', -1))
            return m

        raise NotImplementedError("Unsupported stream type for transform_data_to_model")

    def train_model_from_dataset(self, model: Model, x_file: str, y_file: str) -> MultiClassClassifierMetrics:
        """Train a model using the specified dataset type and files. Returns metrics."""
        if x_file.endswith('.bin'):
            return self.train_model_from_binary_dataset(model, x_file, y_file)
        else:
            return self.train_model_from_original_dataset(model, x_file, y_file)

    def train_model_from_original_dataset(self, model: Model, x_file: str, y_file: str) -> MultiClassClassifierMetrics:
        """Train a model using files `x_file` and `y_file`. Returns metrics."""
        metrics = MultiClassClassifierMetrics()
        metrics.parsingTime = 0
        metrics.trainingTime = 0
        metrics.epochs = self.config.epochs

        if tf is None:
            return metrics

        def _load_csv(path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return np.loadtxt(path, delimiter=',')

        x_path = x_file
        y_path = y_file
        if not os.path.exists(x_path) and os.path.exists(X_TRAIN_PATH):
            x_path = X_TRAIN_PATH
        if not os.path.exists(y_path) and os.path.exists(Y_TRAIN_PATH):
            y_path = Y_TRAIN_PATH

        X = _load_csv(x_path)
        y = _load_csv(y_path)

        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        # Ensure X and y have the same number of samples. If files differ
        # (for example due to trailing newlines or pre-processing issues),
        # trim to the smaller set and log a warning so the caller can
        # investigate. Avoid raising here to keep training robust on-device.
        try:
            n_x = int(X.shape[0])
        except Exception:
            n_x = 0
        try:
            n_y = int(y.shape[0])
        except Exception:
            n_y = 0

        if n_x != n_y:
            _LOG.warning('Data cardinality mismatch: x samples=%s, y samples=%s — trimming to min', n_x, n_y)
            m = min(n_x, n_y)
            if m <= 0:
                _LOG.error('No samples available after trimming (x=%s, y=%s); aborting training', n_x, n_y)
                return metrics
            X = X[:m]
            y = y[:m]

        return self._train_on_data(X, y)

    def build_keras_model(self) -> Any:
        """Compile and cache Keras model based on config."""
        _lazy_import_tf()
        if tf is None:
            return None
        keras = getattr(tf, 'keras', None)
        if keras is None:
            return None

        model_tf = keras.Sequential()
        cfg_layers = getattr(self.config, 'layers', None) or [10, 10]
        arch = [int(x) for x in cfg_layers]
        input_dim = arch[0]
        model_tf.add(keras.Input(shape=(input_dim,)))

        # Map numeric activation codes to Keras activations or special markers
        ACT_MAP = {
            0: 'sigmoid',
            1: 'tanh',
            2: 'relu',
            3: 'leaky',   # special-case: keras.layers.LeakyReLU
            4: 'elu',
            5: 'selu',
            6: 'softmax',
        }

        act_codes = list(getattr(self.config, 'activation_functions', []) or [])
        num_dense = len(arch) - 1
        for i in range(num_dense):
            units = int(arch[i + 1])
            code = act_codes[i] if i < len(act_codes) else 2
            act = ACT_MAP.get(int(code), 'linear')

            if act == 'leaky':
                model_tf.add(keras.layers.Dense(units))
                model_tf.add(keras.layers.LeakyReLU(alpha=0.2))
            else:
                model_tf.add(keras.layers.Dense(units, activation=act))

        # Choose loss: categorical_crossentropy for one-hot targets (classes > 1), else mse
        if arch[-1] > 1:
            loss = 'categorical_crossentropy'
        else:
            loss = 'mse'

        model_tf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss)
        self._last_trained_tf_model = model_tf
        return model_tf

    def set_keras_model_weights(self, keras_model: Any, weights_source: Any) -> bool:
        """Load weights into Keras model's Dense layers from bytes (ESP32 layout) or Model object."""
        _lazy_import_tf()
        if tf is None:
            return False
        keras = getattr(tf, 'keras', None)
        if keras is None:
            return False

        dense_layers = [l for l in keras_model.layers if isinstance(l, keras.layers.Dense)]

        if isinstance(weights_source, (bytes, bytearray)):
            parsed = self._parse_nn_bytes(bytes(weights_source))
            if not parsed:
                _LOG.warning("Failed to parse weights bytes.")
                return False

            if len(dense_layers) != len(parsed['layers']):
                _LOG.warning(
                    "Keras Dense layer count (%d) does not match parsed .nn layer count (%d)",
                    len(dense_layers), len(parsed['layers'])
                )
                return False

            for idx, layer in enumerate(dense_layers):
                parsed_layer = parsed['layers'][idx]
                w = parsed_layer['weights']  # shape: (outputs, inputs)
                b = parsed_layer['biases']   # shape: (outputs,)
                layer.set_weights([w.T, b])
            _LOG.info("Loaded weights from bytes into Keras model layers successfully.")
            return True

        elif isinstance(weights_source, Model):
            biases_flat = list(weights_source.biases or [])
            weights_flat = list(weights_source.weights or [])
            
            b_idx = 0
            w_idx = 0
            for layer in dense_layers:
                curr_w, curr_b = layer.get_weights()
                inputs, outputs = curr_w.shape
                
                b_slice = biases_flat[b_idx : b_idx + outputs]
                b_idx += outputs
                if len(b_slice) < outputs:
                    b_slice = b_slice + [0.0] * (outputs - len(b_slice))
                
                num_elements = inputs * outputs
                w_slice = weights_flat[w_idx : w_idx + num_elements]
                w_idx += num_elements
                if len(w_slice) < num_elements:
                    w_slice = w_slice + [0.0] * (num_elements - len(w_slice))
                
                w_arr = np.array(w_slice, dtype=np.float32).reshape((outputs, inputs)).T
                b_arr = np.array(b_slice, dtype=np.float32)
                layer.set_weights([w_arr, b_arr])
            _LOG.info("Loaded weights from Model object into Keras model layers successfully.")
            return True

        return False

    def _train_on_data(self, X: np.ndarray, y: np.ndarray) -> MultiClassClassifierMetrics:
        """Internal: Train the compiled cached model on X, y using current config."""
        _lazy_import_tf()
        metrics = MultiClassClassifierMetrics()
        metrics.parsingTime = 0
        metrics.trainingTime = 0
        metrics.epochs = self.config.epochs
        metrics.datasetSize = int(X.shape[0])

        if tf is None:
            return metrics
            
        keras = getattr(tf, 'keras', None)
        if keras is None:
            _LOG.debug('TensorFlow keras not available despite tf import; returning placeholder metrics')
            return metrics

        model_tf = self._last_trained_tf_model
        if model_tf is None:
            model_tf = self.build_keras_model()
            if model_tf is None:
                _LOG.error("Failed to build Keras model in _train_on_data")
                return metrics

        if hasattr(tf, 'timestamp'):
            start = tf.timestamp()
        else:
            start = time.time()

        history = model_tf.fit(X, y, epochs=max(1, self.config.epochs), verbose=0)

        if hasattr(tf, 'timestamp'):
            end = tf.timestamp()
        else:
            end = time.time()

        metrics.trainingTime = float(end - start)

        self._last_trained_tf_model = model_tf

        metrics.meanSqrdError = float(history.history.get('loss', [0])[-1])

        # Predictions and classification metrics: convert softmax/prob vectors
        # to class labels via argmax for multi-class, otherwise threshold.
        preds = model_tf.predict(X, verbose=0)
        if preds.ndim > 1 and preds.shape[1] > 1:
            y_pred_labels = np.argmax(preds, axis=1)
        else:
            y_pred_labels = (preds.flatten() >= 0.5).astype(int)

        if y.ndim > 1 and y.shape[1] > 1:
            y_true_labels = np.argmax(y, axis=1)
        else:
            y_true_labels = y.flatten().astype(int)

        n_classes = int(max(y_true_labels.max() if y_true_labels.size > 0 else 0,
                            y_pred_labels.max() if y_pred_labels.size > 0 else 0) + 1)
        metrics.numberOfClasses = n_classes
        metrics.metrics = [ClassClassifierMetrics() for _ in range(n_classes)]

        # per-class confusion counts
        for i in range(y_true_labels.shape[0]):
            t = int(y_true_labels[i])
            p = int(y_pred_labels[i])
            for c in range(n_classes):
                if t == c and p == c:
                    metrics.metrics[c].truePositives += 1
                elif t == c and p != c:
                    metrics.metrics[c].falseNegatives += 1
                elif t != c and p == c:
                    metrics.metrics[c].falsePositives += 1
                else:
                    metrics.metrics[c].trueNegatives += 1

        compute_all_metrics(metrics)
        return metrics

    def export_tflite(self, keras_model: Any, tflite_path: str) -> bool:
        """Convert a Keras model to TFLite and write to `tflite_path`."""
        _lazy_import_tf()
        if tf is None:
            raise RuntimeError("TensorFlow is not available in this environment")
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        dirpath = os.path.dirname(tflite_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if isinstance(tflite_model, (bytes, bytearray)):
            tflite_bytes = bytes(tflite_model)
        else:
            try:
                tflite_bytes = bytes(tflite_model)  # type: ignore[arg-type]
            except Exception:
                _LOG.debug('tflite converter returned non-bytes; writing empty file')
                tflite_bytes = b''
        with open(tflite_path, 'wb') as f:
            f.write(tflite_bytes)
        InterpreterCls: Any = None
        try:
            from tflite_runtime.interpreter import Interpreter as _RTInterpreter  # type: ignore
            InterpreterCls = _RTInterpreter
        except Exception:
            InterpreterCls = None

        if InterpreterCls is None:
            try:
                from tensorflow.lite import Interpreter as _TFInterpreter  # type: ignore
                InterpreterCls = _TFInterpreter
            except Exception:
                InterpreterCls = None

        if InterpreterCls is None:
            try:
                from tensorflow.lite.python.interpreter import Interpreter as _TFPyInterpreter  # type: ignore
                InterpreterCls = _TFPyInterpreter
            except Exception:
                InterpreterCls = None

        if InterpreterCls is not None:
            interp = InterpreterCls(model_path=tflite_path)
            if hasattr(interp, 'allocate_tensors'):
                interp.allocate_tensors()
            try:
                mod_name = getattr(InterpreterCls, '__module__', '')
                if 'tensorflow.lite.python.interpreter' in mod_name:
                    _LOG.info('Using TensorFlow internal TFLite interpreter (module=%s); this may be deprecated', mod_name)
            except Exception:
                pass
            return True

        return True

    def serialize_to_nn_bytes(self, keras_model: Any = None, model: Model | None = None) -> bytes | None:
        """Serialize a trained model to the ESP32 `.nn` binary layout."""
        buf = bytearray()
        if keras_model is not None:
            pairs = []
            for layer in getattr(keras_model, 'layers', []):
                w = layer.get_weights()
                if w and len(w) == 2:
                    kernel, bias = w[0], w[1]
                    pairs.append((kernel, bias))

            num_layers = len(pairs)
            buf += struct.pack('<I', num_layers)
            for kernel, bias in pairs:
                inputs = int(kernel.shape[0])
                outputs = int(kernel.shape[1])
                buf += struct.pack('<I', inputs)
                buf += struct.pack('<I', outputs)
                for j in range(outputs):
                    b = float(bias[j]) if bias is not None else 0.0
                    buf += struct.pack('<f', b)
                    for k in range(inputs):
                        v = float(kernel[k, j])
                        buf += struct.pack('<f', v)
            return bytes(buf)

        if model is not None:
            if not hasattr(self, 'config') or not getattr(self.config, 'layers', None):
                return None
            layers = list(self.config.layers)
            if len(layers) < 2:
                return None
            num_layers = len(layers) - 1
            buf += struct.pack('<I', num_layers)
            biases_flat = list(model.biases or [])
            weights_flat = list(model.weights or [])
            b_idx = 0
            w_idx = 0
            for li in range(num_layers):
                inputs = int(layers[li])
                outputs = int(layers[li + 1])
                buf += struct.pack('<I', inputs)
                buf += struct.pack('<I', outputs)
                for out in range(outputs):
                    b = float(biases_flat[b_idx]) if b_idx < len(biases_flat) else 0.0
                    buf += struct.pack('<f', b)
                    b_idx += 1
                    for inp in range(inputs):
                        v = float(weights_flat[w_idx]) if w_idx < len(weights_flat) else 0.0
                        buf += struct.pack('<f', v)
                        w_idx += 1
            return bytes(buf)

        return None

    def _parse_nn_bytes(self, data: bytes) -> dict | None:
        """Parse an ESP32 .nn binary into a dict with layers: weights and biases."""
        offset = 0
        if len(data) < 4:
            return None
        num_layers = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        layers = []
        for layer_idx in range(num_layers):
            activation = None
            if offset + 1 <= len(data):
                potential_activation = struct.unpack_from('<B', data, offset)[0]
                if 0 <= potential_activation <= 6:
                    if offset + 9 <= len(data):
                        inputs = struct.unpack_from('<I', data, offset+1)[0]
                        outputs = struct.unpack_from('<I', data, offset+5)[0]
                        if 1 <= inputs <= 2000 and 1 <= outputs <= 2000:
                            activation = potential_activation
                            offset += 1
            if offset + 8 > len(data):
                return None
            inputs = struct.unpack_from('<I', data, offset)[0]
            outputs = struct.unpack_from('<I', data, offset+4)[0]
            offset += 8

            remaining = len(data) - offset
            floats_available = remaining // 4
            values_per_output = 1 + inputs
            possible_outputs = floats_available // values_per_output
            read_outputs = min(possible_outputs, outputs)

            biases = np.zeros(outputs, dtype=np.float32)
            weights = np.zeros((outputs, inputs), dtype=np.float32)
            for j in range(read_outputs):
                if offset + 4 > len(data):
                    break
                bias_v = struct.unpack_from('<f', data, offset)[0]
                offset += 4
                biases[j] = bias_v
                for k in range(inputs):
                    if offset + 4 > len(data):
                        break
                    wv = struct.unpack_from('<f', data, offset)[0]
                    offset += 4
                    weights[j, k] = wv
            layers.append({
                'inputs': inputs,
                'outputs': outputs,
                'biases': biases,
                'weights': weights,
                'activation': activation,
            })
        return {'num_layers': num_layers, 'layers': layers}

    def predict_from_current_model(self, model: Model, x):
        """Perform a lightweight predict using the current model."""
        if model is None:
            raise ValueError("model is required")
        length = len(x)
        return [0 for _ in range(length)]

    def train_model_from_binary_dataset(self, model: Model, bin_file: str, meta_file: str) -> MultiClassClassifierMetrics:
        """Train a model reading from a binary dataset (ESP32 format) + metadata JSON."""
        _lazy_import_tf()
        if self._looks_like_juliana_dataset(meta_file):
            return self.train_model_from_juliana_binary_dataset(model, bin_file, meta_file)

        metrics = MultiClassClassifierMetrics()
        metrics.parsingTime = 0
        metrics.trainingTime = 0
        metrics.epochs = self.config.epochs

        if tf is None:
            _LOG.warning("TensorFlow not available; cannot train from binary dataset.")
            return metrics

        if not os.path.exists(bin_file):
            _LOG.error("Binary file not found: %s", bin_file)
            return metrics
        if not os.path.exists(meta_file):
            _LOG.error("Metadata file not found: %s", meta_file)
            return metrics
        
        start_parse = time.time()
        
        # Load Metadata
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        schema = meta.get('schema', [])
        label_col_name = meta.get('label_column', 'activityID')
        label_values = meta.get('label_values', [])
        label_map = meta.get('label_map', None)
        bytes_per_row = meta.get('bytes_per_row', None)
        
        # Calculate row size if not provided
        if bytes_per_row is None:
            bytes_per_row = sum(int(c.get('bytes', 0)) for c in schema)
            
        TYPE_TO_STRUCT = {
            'int8': ('b', 1),
            'uint8': ('B', 1),
            'int32': ('<i', 4),
            'float32': ('<f', 4),
        }
        
        # Prepare parse columns
        parse_cols = []
        label_col_info = None
        timestamp_cols = {'timestamp'}
        
        for c in schema:
            name = c['name']
            t = c['type']
            offset = int(c['offset'])
            b = int(c['bytes'])
            
            if t not in TYPE_TO_STRUCT:
                _LOG.warning("Unsupported type in schema: %s", t)
                continue
                
            fmt = TYPE_TO_STRUCT[t][0]
            parse_cols.append((name, t, offset, b, fmt))
            
            if name == label_col_name:
                label_col_info = (name, t, offset, b, fmt)
            
        if label_col_info is None:
            _LOG.error("Label column '%s' not found in schema.", label_col_name)
            return metrics

        # Parse Binary File
        X_list = []
        y_list = []
        
        with open(bin_file, 'rb') as f:
            while True:
                row = f.read(bytes_per_row)
                if not row or len(row) < bytes_per_row:
                    break
                
                # Parse Label
                l_name, l_type, l_offset, l_bytes, l_fmt = label_col_info
                try:
                    val = struct.unpack_from(l_fmt, row, l_offset)[0]
                except struct.error:
                    val = 0
                label_raw = int(val)
                
                # Handle encoded vs original labels
                label_idx = -1
                if label_map is not None:
                    # Encoded labels: 1-based index (0 = no label)
                    if label_raw == 0:
                        continue # Skip unlabeled
                    label_idx = label_raw - 1
                else:
                    # Original labels: need to map via label_values
                    if label_raw in label_values:
                        try:
                            label_idx = label_values.index(label_raw)
                        except ValueError:
                            continue
                    else:
                        continue # Label not in known values
                
                if label_idx < 0:
                    continue
                
                # Parse Features
                feats = []
                for (name, t, offset, b, fmt) in parse_cols:
                    if name == label_col_name or name in timestamp_cols:
                        continue
                    try:
                        val = struct.unpack_from(fmt, row, offset)[0]
                        feats.append(float(val))
                    except struct.error:
                        feats.append(0.0)
                
                X_list.append(feats)
                y_list.append(label_idx)

        metrics.parsingTime = float(time.time() - start_parse)
        
        n_samples = len(X_list)
        if n_samples == 0:
            _LOG.warning("No valid samples found in binary dataset.")
            return metrics
            
        X = np.array(X_list, dtype=np.float32)
        y_indices = np.array(y_list, dtype=np.int32)
        
        # Convert y to one-hot if needed, based on number of classes
        # Generally _train_on_data expects y to be shaped (N, num_classes) or (N, 1) or (N,)
        # The original code handled one-hot detection.
        # But here we know the class index. Let's make it one-hot to be consistent with
        # typical neural net training if we are doing multi-class.
        
        # Check num classes from model config or data
        # If we have label_values, that dictates num classes.
        if label_values:
            num_classes = len(label_values)
        elif label_map:
             # Assuming label_map is Dict[Original, Encoded]. Max encoded value gives size.
             # but actually just max(y_indices) + 1 is a safe bet for now if dynamic.
            num_classes = max(y_indices) + 1 if len(y_indices) > 0 else 0
        else:
             num_classes = max(y_indices) + 1 if len(y_indices) > 0 else 0
             
        # Create one-hot Y
        if num_classes > 1:
            y_one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
            # Use fancy indexing
            # Clip indices just in case
            y_indices = np.clip(y_indices, 0, num_classes - 1)
            y_one_hot[np.arange(n_samples), y_indices] = 1.0
            y = y_one_hot
        else:
            y = y_indices.reshape((-1, 1)).astype(np.float32)
            
        return self._train_on_data(X, y)

    def train_model_from_juliana_binary_dataset(self, model: Model, bin_file: str, meta_file: str) -> MultiClassClassifierMetrics:
        """Train a model reading the Juliana binary dataset produced from `data_juliana`."""
        _lazy_import_tf()
        metrics = MultiClassClassifierMetrics()
        metrics.parsingTime = 0
        metrics.trainingTime = 0
        metrics.epochs = self.config.epochs

        if tf is None:
            _LOG.warning('TensorFlow not available; cannot train from Juliana binary dataset.')
            return metrics

        if not os.path.exists(bin_file):
            _LOG.error('Binary file not found: %s', bin_file)
            return metrics
        if not os.path.exists(meta_file):
            _LOG.error('Metadata file not found: %s', meta_file)
            return metrics

        start_parse = time.time()

        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        schema = meta.get('schema', [])
        label_col_name = meta.get('label_column', 'ocupada')
        if label_col_name != 'ocupada':
            _LOG.warning("Juliana trainer received unexpected label column '%s'; falling back to generic binary trainer.", label_col_name)
            return self.train_model_from_binary_dataset(model, bin_file, meta_file)

        bytes_per_row = meta.get('bytes_per_row', None)
        if bytes_per_row is None:
            bytes_per_row = sum(int(c.get('bytes', 0)) for c in schema)

        type_to_struct = {
            'int8': ('b', 1),
            'uint8': ('B', 1),
            'int32': ('<i', 4),
            'float32': ('<f', 4),
        }

        parse_cols = []
        label_col_info = None
        for c in schema:
            name = c['name']
            col_type = c['type']
            offset = int(c['offset'])
            bytes_count = int(c['bytes'])
            if col_type not in type_to_struct:
                _LOG.warning('Unsupported type in Juliana schema: %s', col_type)
                continue
            fmt = type_to_struct[col_type][0]
            parse_cols.append((name, col_type, offset, bytes_count, fmt))
            if name == label_col_name:
                label_col_info = (name, col_type, offset, bytes_count, fmt)

        if label_col_info is None:
            _LOG.error("Label column '%s' not found in Juliana schema.", label_col_name)
            return metrics

        x_list = []
        y_list = []

        with open(bin_file, 'rb') as f:
            while True:
                row = f.read(bytes_per_row)
                if not row or len(row) < bytes_per_row:
                    break

                _, label_type, label_offset, _, label_fmt = label_col_info
                try:
                    label_val = struct.unpack_from(label_fmt, row, label_offset)[0]
                except struct.error:
                    label_val = 0

                label_idx = 1 if int(label_val) != 0 else 0

                feats = []
                for name, col_type, offset, _, fmt in parse_cols:
                    if name == label_col_name:
                        continue
                    try:
                        value = struct.unpack_from(fmt, row, offset)[0]
                        feats.append(float(value))
                    except struct.error:
                        feats.append(0.0)

                x_list.append(feats)
                y_list.append(label_idx)

        metrics.parsingTime = float(time.time() - start_parse)

        if not x_list:
            _LOG.warning('No valid samples found in Juliana binary dataset.')
            return metrics

        X = np.array(x_list, dtype=np.float32)
        y_indices = np.array(y_list, dtype=np.int32)

        num_classes = 2
        y = np.zeros((len(y_indices), num_classes), dtype=np.float32)
        y[np.arange(len(y_indices)), np.clip(y_indices, 0, num_classes - 1)] = 1.0

        return self._train_on_data(X, y)

    def _looks_like_juliana_dataset(self, meta_file: str) -> bool:
        try:
            if not os.path.exists(meta_file):
                return False
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return meta.get('label_column') == 'ocupada'
        except Exception:
            _LOG.debug('Failed to inspect metadata file for Juliana dataset', exc_info=True)
            return False



def compute_all_metrics(metrics: MultiClassClassifierMetrics) -> None:
    """Consolidated metrics calculation (Macro, Weighted, and Balanced).
    
    This matches the logic in the ESP32 ModelUtil.h, where 'balanced' metrics
    are support-weighted averages of per-class binary metrics.
    """
    if metrics is None or metrics.metrics is None or metrics.numberOfClasses == 0:
        return

    n = metrics.numberOfClasses
    total_samples = metrics.datasetSize

    precisions = []
    recalls = []
    f1s = []
    accuracies = []
    supports = []

    for c in range(n):
        m = metrics.metrics[c]
        tp = m.truePositives
        fp = m.falsePositives
        tn = m.trueNegatives
        fn = m.falseNegatives
        
        support = tp + fn
        supports.append(support)
        
        # Per-class metrics
        total_class = tp + tn + fp + fn
        acc = (tp + tn) / total_class if total_class > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # 1. Macro Averaged (Simple average of per-class metrics)
    metrics.accuracy = float(sum(accuracies) / n)
    metrics.precision = float(sum(precisions) / n)
    metrics.recall = float(sum(recalls) / n)
    metrics.f1Score = float(sum(f1s) / n)

    # 2. Support-Weighted Metrics
    if total_samples > 0:
        # These are what the ESP32 project calls 'balanced' metrics
        metrics.balancedAccuracy = float(sum(a * s for a, s in zip(accuracies, supports)) / total_samples)
        metrics.balancedPrecision = float(sum(p * s for p, s in zip(precisions, supports)) / total_samples)
        metrics.balancedRecall = float(sum(r * s for r, s in zip(recalls, supports)) / total_samples)
        metrics.balancedF1Score = float(sum(f * s for f, s in zip(f1s, supports)) / total_samples)
    else:
        metrics.balancedAccuracy = metrics.accuracy
        metrics.balancedPrecision = metrics.precision
        metrics.balancedRecall = metrics.recall
        metrics.balancedF1Score = metrics.f1Score

