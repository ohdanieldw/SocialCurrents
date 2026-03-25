"""
Example script showing how to use the multimodal data pipeline.
"""
import os
import argparse
from pathlib import Path

from .pipeline import MultimodalPipeline
from .utils.file_utils import ensure_dir

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Example script for the multimodal data pipeline')
    parser.add_argument('--video', help='Path to a video file')
    parser.add_argument('--audio', help='Path to an audio file')
    parser.add_argument('--data-dir', help='Directory containing video files')
    parser.add_argument('--output-dir', default='output/example', help='Output directory')
    parser.add_argument('--features', default='basic_audio,librosa_spectral',
                       help='Comma-separated list of features to extract')
    parser.add_argument('--save-json', action='store_true',
                    help='Save results to output_dir/pipeline_features.json')

    args = parser.parse_args()
    
    # Convert features string to list
    features = args.features.split(',')
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Initialize pipeline
    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=features,
        device='cpu'  # Use 'cuda' if GPU is available
    )
    
    # Process input based on provided arguments
    if args.video:
        print(f"Processing video file: {args.video}")
        results = pipeline.process_video_file(args.video)

        # Always save JSON (you want guaranteed output)
        import json
        out_path = Path(output_dir) / "pipeline_features.json"
        
        def _json_default(o):
            import numpy as np
            # numpy arrays -> lists
            if isinstance(o, np.ndarray):
                return o.tolist()
            # numpy scalars -> python scalars
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            # bytes -> text
            if isinstance(o, (bytes, bytearray)):
                return o.decode("utf-8", errors="replace")
            # fallback
            return str(o)

        def _json_default(o):
            import numpy as np
            # numpy arrays -> lists
            if isinstance(o, np.ndarray):
                return o.tolist()
            # numpy scalars -> python scalars
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            # bytes -> text
            if isinstance(o, (bytes, bytearray)):
                return o.decode("utf-8", errors="replace")
            # fallback
            return str(o)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)

        print(f"Saved JSON to {out_path}")

        
    elif args.audio:
        print(f"Processing audio file: {args.audio}")
        results = pipeline.process_audio_file(args.audio)
        print(f"Results saved to {output_dir}")
        
    elif args.data_dir:
        print(f"Processing all videos in directory: {args.data_dir}")
        results = pipeline.process_directory(args.data_dir, is_video=True)

        import json
        out_path = Path(output_dir) / "pipeline_features.json"
        with open(out_path, "w", encoding="utf-8") as f:
            
            def _json_default(o):
                import numpy as np
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, (np.float32, np.float64)):
                    return float(o)
                if isinstance(o, (np.int32, np.int64)):
                    return int(o)
                return str(o)

            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)


        print(f"Saved JSON to {out_path}")

        print(f"Processed {len(results)} files. Results saved to {output_dir}")
        
    else:
        # Use default data directory
        default_data_dir = Path(__file__).parent.parent / 'data'
        if default_data_dir.exists() and any(default_data_dir.iterdir()):
            print(f"Processing all videos in default directory: {default_data_dir}")
            results = pipeline.process_directory(default_data_dir, is_video=True)
            print(f"Processed {len(results)} files. Results saved to {output_dir}")
        else:
            print("No input provided and default data directory is empty or doesn't exist.")
            print("Please specify --video, --audio, or --data-dir.")
            return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
