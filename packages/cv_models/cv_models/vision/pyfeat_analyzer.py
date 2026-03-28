"""
Py-Feat Analyzer wrapper to produce pf_* features (AUs, emotions, face geometry).
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from typing import Dict, Any, Optional, Tuple, Iterable, List
import importlib.util
import logging
import sys

import multiprocessing
import shutil
import subprocess
import tempfile
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import cv2

from cv_models.utils.scipy_compat import ensure_legacy_stats

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = (
    "pf_au01",
    "pf_au02",
    "pf_au04",
    "pf_au05",
    "pf_au06",
    "pf_au07",
    "pf_au09",
    "pf_au10",
    "pf_au11",
    "pf_au12",
    "pf_au14",
    "pf_au15",
    "pf_au17",
    "pf_au20",
    "pf_au23",
    "pf_au24",
    "pf_au25",
    "pf_au26",
    "pf_au28",
    "pf_au43",
    "pf_anger",
    "pf_disgust",
    "pf_fear",
    "pf_happiness",
    "pf_sadness",
    "pf_surprise",
    "pf_neutral",
    "pf_facerectx",
    "pf_facerecty",
    "pf_facerectwidth",
    "pf_facerectheight",
    "pf_facescore",
    "pf_pitch",
    "pf_roll",
    "pf_yaw",
    "pf_x",
    "pf_y",
    "pf_z",
)


def _select_central_face(df, frame_width: int) -> 'pd.DataFrame':
    """When multiple faces are detected per image, keep only the most central one.

    Groups rows by the ``input`` column (one group per source image).  For
    groups with a single face the row is kept as-is.  For groups with
    multiple faces the face whose horizontal center is closest to the
    frame's horizontal center is selected.
    """
    if 'input' not in df.columns or len(df) <= 1:
        return df

    frame_cx = frame_width / 2.0
    kept: List[int] = []

    for _input_val, group in df.groupby('input', sort=False):
        if len(group) == 1:
            kept.append(group.index[0])
            continue

        # Compute face center-x for each detected face
        rect_x_col = 'FaceRectX' if 'FaceRectX' in group.columns else 'face_x'
        rect_w_col = 'FaceRectWidth' if 'FaceRectWidth' in group.columns else 'face_w'
        if rect_x_col not in group.columns or rect_w_col not in group.columns:
            # Can't determine position — keep first face
            kept.append(group.index[0])
            continue

        face_cx = pd.to_numeric(group[rect_x_col], errors='coerce') + \
                  pd.to_numeric(group[rect_w_col], errors='coerce') / 2.0
        best_idx = (face_cx - frame_cx).abs().idxmin()
        kept.append(best_idx)
        logging.debug(
            "Py-Feat: %d faces detected in %s, selected face closest to center (idx %s)",
            len(group), _input_val, best_idx,
        )

    return df.loc[kept].reset_index(drop=True)


def _run_detect_images(image_paths: List[str], face_model: str = 'mtcnn',
                       au_model: str = 'svm') -> Optional[List[Dict[str, Any]]]:
    """Run Py-Feat detection in a subprocess-safe function.

    Builds its own Detector, suppresses noisy logging, and returns
    the result as a list of row-dicts so it can cross process boundaries.
    Returns None only if every image fails.
    """
    try:
        import warnings as _w
        _w.filterwarnings("ignore", category=FutureWarning, module="timm")
        _w.filterwarnings("ignore", category=UserWarning, module="sklearn")
        logging.getLogger('feat').setLevel(logging.ERROR)

        # scipy compat patch needed inside subprocess
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from cv_models.utils.scipy_compat import ensure_legacy_stats
            ensure_legacy_stats()
        except Exception:
            pass

        from feat import Detector  # noqa: F811
        detector = Detector(
            device='cpu',
            face_model=face_model,
            landmark_model='mobilefacenet',
            au_model=au_model,
            emotion_model='resmasknet',
            facepose_model='img2pose',
            face_detection_threshold=0.15,
        )

        # Determine frame width for multi-face selection
        frame_width = 1280  # fallback
        try:
            img = cv2.imread(image_paths[0])
            if img is not None:
                frame_width = img.shape[1]
        except Exception:
            pass

        # detect_images (plural) may not exist in all py-feat versions
        try:
            result = detector.detect_images(image_paths)
        except (AttributeError, TypeError):
            # Fall back to per-image detection
            results_list = []
            for p in image_paths:
                try:
                    r = detector.detect_image(p)
                    results_list.append(r)
                except Exception as e:
                    logging.warning("Py-Feat: detect_image failed on %s: %s", p, e)
            result = results_list if results_list else None

        if result is None:
            return None

        # Serialise to list-of-dicts so it can be returned through the Pool
        if isinstance(result, list):
            rows = []
            for single in result:
                df = single.to_pandas() if hasattr(single, 'to_pandas') else single
                if df is not None and len(df) > 0:
                    df = _select_central_face(df, frame_width)
                    rows.append(df.to_dict(orient='records'))
                else:
                    rows.append([])
            return rows

        df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        if df is not None and len(df) > 0:
            df = _select_central_face(df, frame_width)
            return [df.to_dict(orient='records')]
        return None
    except Exception as e:
        logging.warning("Py-Feat subprocess error: %s", e)
        return None


class PyFeatAnalyzer:
    MAX_SAMPLE_FRAMES = 120
    BATCH_SIZE = 12
    MAX_FRAME_EDGE = 1280

    def __init__(self, device: str = 'cpu', sample_rate: int = 5, batch_timeout: float = 30,
                 face_model: str = 'mtcnn', au_model: str = 'svm'):
        self.device = device
        self.sample_rate = max(1, int(sample_rate))
        self.batch_timeout = float(batch_timeout) if batch_timeout else None
        self.face_model = face_model
        self.au_model = au_model
        self._init_error: Optional[str] = None
        ensure_legacy_stats()
        self._ensure_lib2to3()

    def _build_detector(self):
        """Instantiate a fresh Py-Feat Detector with compatibility patches applied."""
        try:
            self._patch_sklearn_compat()

            # Patch torch.nn.Module.load_state_dict to handle DataParallel mismatch
            import torch.nn as nn  # type: ignore[import]
            original_load_state_dict = nn.Module.load_state_dict

            def patched_load_state_dict(module_self, state_dict, strict=True):
                """Patched load_state_dict that handles DataParallel prefix mismatch."""
                if isinstance(state_dict, dict) and state_dict:
                    model_keys = set(module_self.state_dict().keys())
                    state_keys = set(state_dict.keys())

                    model_has_module = any(k.startswith('module.') for k in model_keys)
                    state_has_module = any(k.startswith('module.') for k in state_keys)

                    if model_has_module and not state_has_module:
                        logger.info("Adding 'module.' prefix to state_dict keys")
                        state_dict = {'module.' + k: v for k, v in state_dict.items()}
                    elif not model_has_module and state_has_module:
                        logger.info("Removing 'module.' prefix from state_dict keys")
                        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

                return original_load_state_dict(module_self, state_dict, strict=strict)

            nn.Module.load_state_dict = patched_load_state_dict

            try:
                from feat import Detector  # type: ignore
                # Lower the face detection threshold to improve sensitivity on small/low-contrast faces
                detector = Detector(
                    device=self.device,
                    face_model=self.face_model,
                    landmark_model='mobilefacenet',
                    au_model=self.au_model,
                    emotion_model='resmasknet',
                    facepose_model='img2pose',
                    face_detection_threshold=0.15,
                )
                self._init_error = None
            finally:
                # Restore original method even if Detector() fails
                nn.Module.load_state_dict = original_load_state_dict
                self._restore_sklearn_dtype()

            return detector
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("Py-Feat detector init failed: %s", exc)
            self._restore_sklearn_dtype()
            return None

    @staticmethod
    def _release_detector(detector) -> None:
        """Release resources tied to a Detector instance to reduce peak memory."""
        if detector is None:
            return
        try:
            # Explicitly drop heavy references that keep torch models alive
            for attr in ["face_detector", "au_model", "emotion_model", "landmark_model"]:
                if hasattr(detector, attr):
                    setattr(detector, attr, None)
        finally:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    def _postprocess_df(self, df) -> Dict[str, Any]:
        stats: Dict[str, float] = {}
        if hasattr(df, 'columns'):
            for col in df.columns:
                values = df[col].to_numpy(dtype=float, copy=False)
                if values.size == 0:
                    continue
                valid = values[np.isfinite(values)]
                if valid.size == 0:
                    continue
                stats[col] = float(np.nanmean(valid))
        return self._postprocess_stats(stats)

    def _postprocess_stats(self, stats: Dict[str, float]) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        au_map = {
            'pf_au01': ('AU01_r', 'AU01'), 'pf_au02': ('AU02_r', 'AU02'), 'pf_au04': ('AU04_r', 'AU04'),
            'pf_au05': ('AU05_r', 'AU05'), 'pf_au06': ('AU06_r', 'AU06'), 'pf_au07': ('AU07_r', 'AU07'),
            'pf_au09': ('AU09_r', 'AU09'), 'pf_au10': ('AU10_r', 'AU10'), 'pf_au11': ('AU11_r', 'AU11'),
            'pf_au12': ('AU12_r', 'AU12'), 'pf_au14': ('AU14_r', 'AU14'), 'pf_au15': ('AU15_r', 'AU15'),
            'pf_au17': ('AU17_r', 'AU17'), 'pf_au20': ('AU20_r', 'AU20'), 'pf_au23': ('AU23_r', 'AU23'),
            'pf_au24': ('AU24_r', 'AU24'), 'pf_au25': ('AU25_r', 'AU25'), 'pf_au26': ('AU26_r', 'AU26'),
            'pf_au28': ('AU28_r', 'AU28'), 'pf_au43': ('AU43_r', 'AU43')
        }
        for out_key, candidates in au_map.items():
            for col in candidates:
                if col in stats:
                    features[out_key] = float(stats[col])
                    break

        emo_cols = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
        for emo in emo_cols:
            if emo in stats:
                features[f'pf_{emo}'] = float(stats[emo])

        geometry_map = {
            'pf_facerectx': ('face_x', 'FaceRectX'),
            'pf_facerecty': ('face_y', 'FaceRectY'),
            'pf_facerectwidth': ('face_w', 'FaceRectWidth'),
            'pf_facerectheight': ('face_h', 'FaceRectHeight'),
            'pf_facescore': ('face_score', 'FaceScore'),
        }
        for out_key, candidates in geometry_map.items():
            for col in candidates:
                if col in stats:
                    features[out_key] = float(stats[col])
                    break

        angle_map = {
            'pf_pitch': ('pitch', 'Pitch'),
            'pf_roll': ('roll', 'Roll'),
            'pf_yaw': ('yaw', 'Yaw'),
        }
        for out_key, candidates in angle_map.items():
            for col in candidates:
                if col in stats:
                    features[out_key] = float(stats[col])
                    break

        x_cols = [stats[col] for col in stats if col.startswith('x_') and np.isfinite(stats[col])]
        y_cols = [stats[col] for col in stats if col.startswith('y_') and np.isfinite(stats[col])]
        z_cols = [stats[col] for col in stats if col.startswith('z_') and np.isfinite(stats[col])]
        if x_cols:
            features['pf_x'] = float(np.mean(x_cols))
        if y_cols:
            features['pf_y'] = float(np.mean(y_cols))
        if z_cols:
            features['pf_z'] = float(np.mean(z_cols))
        return features

    @staticmethod
    def _ensure_required_features(features: Dict[str, Any]) -> Dict[str, Any]:
        for key in REQUIRED_FEATURES:
            features.setdefault(key, 0.0)
        return features

    def get_feature_dict(self, video_path: str, keep_per_frame: bool = False) -> Dict[str, Any]:
        detector = self._build_detector()

        if detector is None:
            message = self._init_error or "Py-Feat is not available in this environment."
            logger.error(f"Py-Feat detector unavailable: {message}")
            features = self._ensure_required_features({})
            features["pf_error"] = message
            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": features
                }
            }

        cleanup_dir: Optional[Path] = None
        try:
            try:
                feature_values = self._detect_features(detector, video_path, keep_per_frame=keep_per_frame)
            except Exception as first_error:
                logger.warning("Py-Feat detection failed (%s); attempting sanitized copy", first_error)
                sanitized_path, cleanup_dir = self._sanitize_video(video_path)
                if sanitized_path is None:
                    raise

                try:
                    feature_values = self._detect_features(detector, sanitized_path, keep_per_frame=keep_per_frame)
                except Exception as sanitized_error:
                    raise RuntimeError(f"Py-Feat sanitized detection failed: {sanitized_error}") from first_error

            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": feature_values
                }
            }
        except Exception as e:
            logger.error(f"Py-Feat analysis failed: {e}")
            features = self._ensure_required_features({})
            features["pf_error"] = str(e)
            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": features
                }
            }
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            self._release_detector(detector)

    @staticmethod
    def _patch_sklearn_compat():
        """Patch sklearn tree loading to handle old pickle format without missing_go_to_left field."""
        try:
            from sklearn.tree import _tree
            import numpy as np
            
            # Check if already patched - if so, just ensure it's still the old dtype
            if hasattr(_tree, '_patched_for_old_pickles'):
                # Already patched, ensure dtype is correct
                if not hasattr(_tree, '_original_NODE_DTYPE'):
                    # Restore original if lost
                    _tree._original_NODE_DTYPE = np.dtype([
                        ('left_child', '<i8'),
                        ('right_child', '<i8'),
                        ('feature', '<i8'),
                        ('threshold', '<f8'),
                        ('impurity', '<f8'),
                        ('n_node_samples', '<i8'),
                        ('weighted_n_node_samples', '<f8'),
                        ('missing_go_to_left', 'u1')
                    ])
                
                # Re-apply old dtype in case it was restored
                old_node_dtype = np.dtype([
                    ('left_child', '<i8'),
                    ('right_child', '<i8'),
                    ('feature', '<i8'),
                    ('threshold', '<f8'),
                    ('impurity', '<f8'),
                    ('n_node_samples', '<i8'),
                    ('weighted_n_node_samples', '<f8')
                ])
                _tree.NODE_DTYPE = old_node_dtype
                logger.debug("Re-applied sklearn tree dtype patch")
                return
            
            logger.info("Patching sklearn.tree._tree for old pickle compatibility")
            
            # Save original NODE_DTYPE
            _tree._original_NODE_DTYPE = _tree.NODE_DTYPE
            
            # Create a version without missing_go_to_left for backward compatibility
            # This matches the old format that py-feat models were saved with
            old_node_dtype = np.dtype([
                ('left_child', '<i8'),
                ('right_child', '<i8'),
                ('feature', '<i8'),
                ('threshold', '<f8'),
                ('impurity', '<f8'),
                ('n_node_samples', '<i8'),
                ('weighted_n_node_samples', '<f8')
            ])
            
            # Patch the module's dtype temporarily
            _tree.NODE_DTYPE = old_node_dtype
            
            # Mark as patched
            _tree._patched_for_old_pickles = True
            
            logger.info("sklearn tree dtype patched successfully")
            
        except Exception as e:
            logger.warning(f"Could not patch sklearn tree compatibility: {e}")
    
    @staticmethod
    def _restore_sklearn_dtype():
        """Restore original sklearn tree dtype after Py-Feat initialization."""
        try:
            from sklearn.tree import _tree
            
            if hasattr(_tree, '_original_NODE_DTYPE'):
                logger.info("Restoring original sklearn tree dtype")
                _tree.NODE_DTYPE = _tree._original_NODE_DTYPE
                delattr(_tree, '_original_NODE_DTYPE')
                if hasattr(_tree, '_patched_for_old_pickles'):
                    delattr(_tree, '_patched_for_old_pickles')
                    
        except Exception as e:
            logger.warning(f"Could not restore sklearn tree dtype: {e}")

    @staticmethod
    def _ensure_lib2to3() -> None:
        """Ensure lib2to3 (removed in Python 3.13) is available for Py-Feat imports."""
        spec = importlib.util.find_spec("lib2to3")
        if spec is not None:
            return

        from types import ModuleType

        shim = ModuleType("lib2to3")
        shim.__path__ = []  # type: ignore[attr-defined]

        pytree = ModuleType("lib2to3.pytree")

        def _convert(node, results=None):  # pragma: no cover - compatibility shim
            return node

        pytree.convert = _convert  # type: ignore[attr-defined]

        shim.pytree = pytree  # type: ignore[attr-defined]

        sys.modules.setdefault("lib2to3", shim)
        sys.modules.setdefault("lib2to3.pytree", pytree)

    def _detect_features(self, detector, video_path: str, keep_per_frame: bool = False) -> Dict[str, Any]:
        """Run Py-Feat detection on a sampled subset of frames to bound memory usage."""
        frame_indices = self._select_frame_indices(video_path)
        if not frame_indices:
            logger.warning("Py-Feat: no readable frames detected in %s", video_path)
            return self._ensure_required_features({})

        # Get FPS for per-frame timestamps
        _cap = cv2.VideoCapture(video_path)
        fps = _cap.get(cv2.CAP_PROP_FPS) if _cap.isOpened() else 30.0
        _cap.release()
        if not fps or fps <= 0:
            fps = 30.0

        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        per_frame_records: List[Dict[str, Any]] = []

        logged_columns = False

        def update_stats(result_df, batch_frame_indices: Optional[List[int]] = None) -> None:
            if result_df is None or len(getattr(result_df, 'index', [])) == 0:
                return

            df = result_df.to_pandas() if hasattr(result_df, 'to_pandas') else result_df
            if df is None or len(df) == 0:
                return

            nonlocal logged_columns
            if not logged_columns:
                logger.info("Py-Feat result columns: %s", list(df.columns))
                logged_columns = True

            # Aggregate means incrementally without storing large DataFrames
            for col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                if series.isna().all():
                    continue
                values = series.to_numpy(dtype=float, copy=False)
                if values.size == 0:
                    continue
                valid = values[np.isfinite(values)]
                if valid.size == 0:
                    continue
                totals[col] = totals.get(col, 0.0) + float(np.nansum(valid))
                counts[col] = counts.get(col, 0) + int(valid.size)

            # Collect per-frame records when requested
            if keep_per_frame and batch_frame_indices is not None:
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= len(batch_frame_indices):
                        break
                    frame_idx = batch_frame_indices[i]
                    row_stats = {}
                    for col in df.columns:
                        val = pd.to_numeric(row[col], errors='coerce')
                        if pd.notna(val) and np.isfinite(float(val)):
                            row_stats[col] = float(val)
                    feat = self._postprocess_stats(row_stats)
                    feat['frame_idx'] = frame_idx
                    feat['timestamp'] = frame_idx / fps
                    per_frame_records.append(feat)

        self._patch_sklearn_compat()
        try:
            for batch_num, batch in enumerate(self._batched_frames(video_path, frame_indices, self.BATCH_SIZE)):
                if not batch:
                    continue

                batch_frame_idx_list = [fidx for fidx, _ in batch]

                temp_dir = tempfile.TemporaryDirectory(prefix="pyfeat_batch_")
                try:
                    image_paths: List[str] = []
                    for frame_idx, rgb_frame in batch:
                        frame_path = Path(temp_dir.name) / f"frame_{frame_idx:06d}_{batch_num:03d}.png"
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(str(frame_path), bgr_frame):
                            continue
                        image_paths.append(str(frame_path))

                    if not image_paths:
                        continue

                    # Run detection in a subprocess so it can be killed on timeout.
                    # detect_images() can block in C/PyTorch code that ignores signals.
                    pool = multiprocessing.Pool(1)
                    try:
                        async_result = pool.apply_async(_run_detect_images, (image_paths, self.face_model, self.au_model))
                        try:
                            serialized = async_result.get(timeout=self.batch_timeout)
                        except multiprocessing.TimeoutError:
                            logger.warning(
                                "Py-Feat: batch %d timed out after %ds (frames %s), skipping batch",
                                batch_num, int(self.batch_timeout), batch_frame_idx_list,
                            )
                            pool.terminate()
                            pool.join()
                            continue
                    finally:
                        pool.terminate()
                        pool.join()

                    # Deserialise the subprocess result and feed into update_stats
                    if serialized is not None:
                        for records in serialized:
                            if records:
                                df = pd.DataFrame(records)
                                # Figure out which frame indices this chunk covers
                                chunk_len = len(df)
                                chunk_indices = batch_frame_idx_list[:chunk_len]
                                update_stats(df, chunk_indices)

                finally:
                    temp_dir.cleanup()
        finally:
            self._restore_sklearn_dtype()

        aggregated = self._finalize_means(totals, counts)
        # If aggregation produced no stats, warn once at the video level so it's clear
        # why all pf_* features are zero — avoids noisy repeated warnings from the
        # underlying library while still surfacing the issue to users.
        if not aggregated:
            logger.warning("Py-Feat: no face detected in sampled frames for %s", video_path)
        features = self._postprocess_stats(aggregated)
        features = self._ensure_required_features(features)
        if keep_per_frame and per_frame_records:
            features['pf_per_frame'] = per_frame_records
        return features

    @staticmethod
    def _finalize_means(totals: Dict[str, float], counts: Dict[str, int]) -> Dict[str, Any]:
        if not totals:
            return {}
        return {
            col: totals[col] / counts[col]
            for col in totals
            if counts.get(col, 0) > 0
        }

    def _select_frame_indices(self, video_path: str) -> List[int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if total_frames <= 0:
            return []

        indices = list(range(0, total_frames, self.sample_rate))
        return indices

    @staticmethod
    def _batched_frames(video_path: str, indices: Iterable[int], batch_size: int) -> Iterable[List[Tuple[int, np.ndarray]]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        batch: List[Tuple[int, np.ndarray]] = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if not success or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                max_dim = max(height, width)
                if max_dim > PyFeatAnalyzer.MAX_FRAME_EDGE:
                    scale = PyFeatAnalyzer.MAX_FRAME_EDGE / float(max_dim)
                    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                    rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
                batch.append((idx, rgb))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch
        finally:
            cap.release()

    def _sanitize_video(self, video_path: str) -> Tuple[Optional[str], Optional[Path]]:
        """Create a sanitized temporary copy of the video tolerant to decode errors."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            logger.warning("FFmpeg not found; cannot sanitize video for Py-Feat.")
            return None, None

        source_path = Path(video_path)
        if not source_path.exists():
            logger.warning("Video path does not exist: %s", video_path)
            return None, None

        temp_dir = Path(tempfile.mkdtemp(prefix="pyfeat_sanitize_"))
        sanitized_path = temp_dir / source_path.name

        cmd = [
            ffmpeg_path,
            "-y",
            "-err_detect",
            "ignore_err",
            "-i",
            str(source_path),
            "-c",
            "copy",
            str(sanitized_path),
        ]

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.stderr:
                logger.debug("FFmpeg sanitize stderr: %s", completed.stderr.decode(errors="ignore"))
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "FFmpeg failed to sanitize video for Py-Feat: %s",
                exc.stderr.decode(errors="ignore") if exc.stderr else exc,
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None
        except Exception as exc:
            logger.warning("Unexpected error while sanitizing video for Py-Feat: %s", exc)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None

        logger.info("Created sanitized video copy for Py-Feat at %s", sanitized_path)
        return str(sanitized_path), temp_dir
