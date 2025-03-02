# Standard library imports
import os
import io
import gc
import re
import time
import asyncio
import traceback
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import threading
import concurrent.futures
from functools import partial
import json
import configparser
import logging
import uuid
import zipfile
from datetime import datetime
from queue import Queue
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Third-party library imports
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import spandrel
import spandrel_extra_arches
from flask import Flask, request, jsonify, render_template, send_file, Response, url_for
from werkzeug.utils import secure_filename

# Local module imports
from utils.vram_estimator import estimate_vram_and_tile_size, get_free_vram
from utils.fuzzy_model_matcher import find_closest_models, search_models
from utils.alpha_handler import handle_alpha
from utils.resize_module import resize_image, get_available_filters, MIN_SCALE_FACTOR, MAX_SCALE_FACTOR
from utils.image_info import get_image_info

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Print CUDA availability information
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: CUDA is not available. Processing will use CPU only.")

# Install extra architectures
spandrel_extra_arches.install()

app = Flask(__name__)
# Set a very large max content length (e.g., 16GB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Global variables
models = {}
thread_pool = ThreadPoolExecutor(max_workers=int(config['Processing'].get('ThreadPoolWorkers', 1)))
last_cleanup_time = time.time()
CLEANUP_INTERVAL = 3 * 60 * 60  # 3 hours in seconds

# Batch processing variables
batch_jobs = {}
batch_queue = Queue()
batch_thread = None
batch_results_dir = 'batch_results'
os.makedirs(batch_results_dir, exist_ok=True)

# Configuration management
def load_config():
    config.read('config.ini')
    return {section: dict(config[section]) for section in config.sections()}

def save_config(new_config):
    for section in new_config:
        if section not in config:
            config.add_section(section)
        for key, value in new_config[section].items():
            config[section][key] = str(value)
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Model management
def load_model(model_name):
    if model_name in models:
        return models[model_name]
    
    model_path = os.path.join(config['Paths']['ModelPath'], f"{model_name}")
    if os.path.exists(model_path + '.pth'):
        model_path = model_path + '.pth'
    elif os.path.exists(model_path + '.safetensors'):
        model_path = model_path + '.safetensors'
    else:
        raise ValueError(f"Model file not found: {model_name}")
    
    try:
        model = spandrel.ModelLoader().load_from_file(model_path)
        if isinstance(model, spandrel.ImageModelDescriptor):
            # Force model to CUDA if available
            if torch.cuda.is_available():
                models[model_name] = model.cuda().eval()
                logger.info(f"Model {model_name} loaded on CUDA")
            else:
                # Fallback to CPU
                logger.warning(f"Using CPU for model {model_name} - this will be much slower!")
                models[model_name] = model.cpu().eval()
            return models[model_name]
        else:
            raise ValueError(f"Invalid model type for {model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def list_available_models():
    model_path = config['Paths']['ModelPath']
    return [os.path.splitext(f)[0] for f in os.listdir(model_path) 
            if f.endswith(('.pth', '.safetensors'))]

# Image processing functions
def upscale_image(image, model, tile_size, alpha_handling, has_alpha, precision, check_cancelled):
    try:
        # Add diagnostic info for CUDA usage
        if hasattr(model, 'parameters'):
            device = next(model.parameters()).device
            logger.debug(f"Starting upscale with model on device: {device}")
        
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        logger.debug(f"Tile size: {tile_size}, precision: {precision}")
        
        def upscale_func(img):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            # Explicitly move tensor to CUDA if available
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            _, _, h, w = img_tensor.shape
            output_h, output_w = h * model.scale, w * model.scale
            logger.debug(f"Processing image: {w}x{h} -> {output_w}x{output_h}")

            if model.supports_bfloat16 and precision in ['auto', 'bf16']:
                output_dtype = torch.bfloat16
                autocast_dtype = torch.bfloat16
            elif model.supports_half and precision in ['auto', 'fp16']:
                output_dtype = torch.float16
                autocast_dtype = torch.float16
            else:
                output_dtype = torch.float32
                autocast_dtype = None

            logger.debug(f"Using precision mode: {autocast_dtype}")

            # Set device according to CUDA availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            output_tensor = torch.zeros((1, img_tensor.shape[1], output_h, output_w), 
                                     dtype=output_dtype, device=device)

            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    if check_cancelled():
                        raise asyncio.CancelledError("Upscale operation was cancelled")

                    tile = img_tensor[:, :, y:min(y+tile_size, h), x:min(x+tile_size, w)]
                    with torch.inference_mode():
                        if autocast_dtype:
                            with torch.autocast(device_type=device, dtype=autocast_dtype):
                                upscaled_tile = model(tile)
                        else:
                            upscaled_tile = model(tile)
                    
                    output_tensor[:, :, 
                                y*model.scale:min((y+tile_size)*model.scale, output_h),
                                x*model.scale:min((x+tile_size)*model.scale, output_w)].copy_(upscaled_tile)

            return Image.fromarray((output_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))

        if has_alpha:
            return handle_alpha(image, upscale_func, alpha_handling, 
                              config['Processing'].getboolean('GammaCorrection', False))
        else:
            return upscale_func(image)

    except Exception as e:
        logger.error(f"Error in upscale_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_image_metadata(image):
    """Get metadata for an image object."""
    info = {
        'format': image.format,
        'mode': image.mode,
        'size': {'width': image.width, 'height': image.height},
        'pixels': image.width * image.height,
        'megapixels': (image.width * image.height) / 1_000_000
    }
    
    # Add additional metadata if available
    if hasattr(image, 'info'):
        if 'dpi' in image.info:
            info['dpi'] = image.info['dpi']
        if 'icc_profile' in image.info:
            info['has_icc_profile'] = True
        if 'exif' in image.info:
            info['has_exif'] = True
    
    # Check for alpha channel
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        info['has_alpha'] = True
    else:
        info['has_alpha'] = False
    
    # Estimate memory usage
    estimated_memory = (image.width * image.height * len(image.getbands()) * 8) / (8 * 1024 * 1024)  # in MB
    info['estimated_memory'] = f"{estimated_memory:.2f} MB"
    
    return info

# Batch processing functions
def process_batch_queue():
    """Process the batch queue in a separate thread with parallel processing."""
    while True:
        try:
            job_id = batch_queue.get()
            if job_id is None:  # Shutdown signal
                break
                
            job = batch_jobs[job_id]
            job['status'] = 'processing'
            
            # Get the output path from the job or use the default
            output_path = job.get('output_path', config['Paths'].get('DefaultOutputPath', 'output'))
            
            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            try:
                # Load the model once for the entire batch
                model = load_model(job['model_name'])
                
                # Explicitly ensure model is on CUDA
                if torch.cuda.is_available():
                    model = model.cuda()
                    logger.info(f"Model for job {job_id} loaded on CUDA device")
                else:
                    logger.warning(f"CUDA not available! Job {job_id} will process on CPU (much slower)")
                
                # Determine number of images to process in parallel
                parallel_size = int(config['Processing'].get('BatchParallelSize', 4))
                logger.info(f"Processing job {job_id} with parallel size: {parallel_size}")
                
                # Process files in parallel
                total_count = len(job['files'])
                processed_count = 0
                completed_count = 0
                failed_count = 0
                
                # Lock for updating shared state
                progress_lock = threading.Lock()
                
                # Process a single image
                def process_image(file_info):
                    nonlocal processed_count, completed_count, failed_count
                    
                    if job.get('cancelled', False) or file_info['status'] != 'pending':
                        return
                    
                    # Mark as processing
                    with progress_lock:
                        file_info['status'] = 'processing'
                        job['current_file'] = file_info['filename']
                    
                    try:
                        # Get the file buffer from memory
                        file_buffer = file_info['buffer']
                        file_buffer.seek(0)  # Reset pointer to beginning
                        
                        # Open the image from buffer
                        image = Image.open(file_buffer)
                        
                        # Convert RGBA images to RGB if alpha handling is 'discard'
                        if job['alpha_handling'] == 'discard' and image.mode == 'RGBA':
                            image = image.convert('RGB')
                        
                        has_alpha = image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)
                        
                        # Calculate tile size and VRAM
                        input_size = (image.width, image.height)
                        estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(
                            model=model,
                            input_size=input_size
                        )
                        
                        logger.info(f"Processing image {file_info['filename']} with CUDA: {torch.cuda.is_available()}")
                        
                        # Process image
                        result = upscale_image(
                            image=image,
                            model=model,
                            tile_size=adjusted_tile_size,
                            alpha_handling=job['alpha_handling'],
                            has_alpha=has_alpha,
                            precision=config['Processing'].get('Precision', 'auto').lower(),
                            check_cancelled=lambda: job.get('cancelled', False)
                        )
                        
                        # Save the result directly to the user-specified output directory
                        output_filename = f"upscaled_{file_info['filename']}"
                        user_output_path = os.path.join(output_path, output_filename)
                        result.save(user_output_path, format='PNG')
                        
                        with progress_lock:
                            file_info['status'] = 'completed'
                            file_info['output_path'] = user_output_path
                            completed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing file {file_info['filename']}: {str(e)}")
                        with progress_lock:
                            file_info['status'] = 'failed'
                            file_info['error'] = str(e)
                            failed_count += 1
                    
                    finally:
                        # Clean up memory
                        if 'buffer' in file_info:
                            file_info['buffer'] = None  # Allow buffer to be garbage collected
                        
                        # Local GPU cleanup for this thread
                        torch.cuda.empty_cache()
                    
                    # Update progress
                    with progress_lock:
                        nonlocal processed_count
                        processed_count += 1
                        job['progress'] = (processed_count / total_count) * 100
                        job['current_file_index'] = processed_count
                        job['completed_count'] = completed_count
                        job['failed_count'] = failed_count
                
                # Create batches of files to process
                pending_files = [f for f in job['files'] if f['status'] == 'pending']
                
                # Process in parallel with a thread pool
                with ThreadPoolExecutor(max_workers=parallel_size) as executor:
                    # Submit all tasks
                    futures = [executor.submit(process_image, file_info) for file_info in pending_files]
                    
                    # Wait for all tasks to complete
                    for future in concurrent.futures.as_completed(futures):
                        if job.get('cancelled', False):
                            break
                        # Get any exceptions
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {str(e)}")
                
                # If all images were processed (none were cancelled)
                if not job.get('cancelled', False):
                    job['status'] = 'completed'
                    
                # Final cleanup of GPU memory
                models.clear()
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch job {job_id}: {str(e)}")
                job['status'] = 'failed'
                job['error'] = str(e)
            
            finally:
                job['end_time'] = time.time()
                job['final_output_path'] = output_path  # Store the final output path
                batch_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in batch processing thread: {str(e)}")
            logger.error(traceback.format_exc())

def start_batch_thread():
    """Start the batch processing thread if not already running."""
    global batch_thread
    if batch_thread is None or not batch_thread.is_alive():
        batch_thread = threading.Thread(target=process_batch_queue, daemon=True)
        batch_thread.start()

# Routes
@app.route('/')
def index():
    return render_template('index.html', 
                         models=list_available_models(),
                         config=load_config())

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify(load_config())
    else:
        new_config = request.json
        save_config(new_config)
        return jsonify({"status": "success"})

@app.route('/api/models')
def get_models():
    return jsonify(list_available_models())

@app.route('/api/select-output-path', methods=['GET'])
def select_output_path():
    """Open a file dialog to select output directory (server-side)."""
    try:
        # Use tkinter to create a file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Show the directory selection dialog
        current_path = config['Paths'].get('DefaultOutputPath', os.getcwd())
        selected_path = filedialog.askdirectory(
            initialdir=current_path,
            title="Select Output Directory"
        )
        
        # If user cancels, return the current path
        if not selected_path:
            return jsonify({"path": current_path})
        
        # Update the config with the new path
        config['Paths']['DefaultOutputPath'] = selected_path
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
            
        return jsonify({"path": selected_path})
        
    except Exception as e:
        logger.error(f"Error selecting output path: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upscale', methods=['POST'])
def upscale():
    try:
        logger.debug("Starting upscale request")
        
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        model_name = request.form.get('model')
        alpha_handling = request.form.get('alpha_handling', 
                                        config['Processing']['DefaultAlphaHandling'])
        
        if not model_name:
            return jsonify({"error": "No model specified"}), 400
        
        logger.debug(f"Processing upscale request: model={model_name}, alpha={alpha_handling}")
        
        # Load and process image
        image = Image.open(file.stream)
        
        # Convert RGBA images to RGB if alpha handling is 'discard'
        if alpha_handling == 'discard' and image.mode == 'RGBA':
            image = image.convert('RGB')
        
        has_alpha = image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)
        
        # Load model
        try:
            model = load_model(model_name)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
        
        # Calculate tile size and VRAM
        try:
            input_size = (image.width, image.height)
            estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(
                model=model,
                input_size=input_size
            )
            
            logger.debug(f"VRAM estimation complete. Tile size: {adjusted_tile_size}")
            
        except Exception as e:
            logger.error(f"Error estimating VRAM: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to estimate VRAM requirements: {str(e)}"}), 500
        
        # Process image
        try:
            result = upscale_image(
                image=image,
                model=model,
                tile_size=adjusted_tile_size,
                alpha_handling=alpha_handling,
                has_alpha=has_alpha,
                precision=config['Processing'].get('Precision', 'auto').lower(),
                check_cancelled=lambda: False
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            return jsonify({"error": "Not enough GPU memory to process the image. Try a smaller image or different model."}), 500
        except Exception as e:
            logger.error(f"Error during upscaling: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to upscale image: {str(e)}"}), 500
        
        # Save result
        try:
            output = io.BytesIO()
            save_format = 'PNG'
            result.save(output, format=save_format)
            output.seek(0)
            
            return send_file(
                output,
                mimetype='image/png',
                as_attachment=True,
                download_name=f"{file.filename}"
            )
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            return jsonify({"error": f"Failed to save result: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error in upscale endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

@app.route('/api/batch-upscale', methods=['POST'])
def batch_upscale():
    try:
        logger.debug("Starting batch upscale request")
        
        if 'images[]' not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist('images[]')
        model_name = request.form.get('model')
        alpha_handling = request.form.get('alpha_handling', 
                                        config['Processing']['DefaultAlphaHandling'])
        output_path = request.form.get('output_path')
        
        if not model_name:
            return jsonify({"error": "No model specified"}), 400
        
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        # If no output path provided, use the default
        if not output_path:
            output_path = config['Paths'].get('DefaultOutputPath', 'output')
        
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        logger.debug(f"Processing batch upscale request: model={model_name}, alpha={alpha_handling}, files={len(files)}, output={output_path}")
        
        # Create a new batch job ID
        job_id = str(uuid.uuid4())
        
        # Create in-memory file information
        file_info_list = []
        for file in files:
            if file.filename:
                # Read the file into memory
                file_data = file.read()
                file_buffer = BytesIO(file_data)
                
                filename = secure_filename(file.filename)
                
                file_info = {
                    'filename': filename,
                    'buffer': file_buffer,  # Store file in memory
                    'status': 'pending',
                    'size': len(file_data)
                }
                file_info_list.append(file_info)
        
        # Create the job data structure
        job = {
            'id': job_id,
            'status': 'queued',
            'model_name': model_name,
            'alpha_handling': alpha_handling,
            'files': file_info_list,
            'total_files': len(file_info_list),
            'progress': 0,
            'current_file': None,
            'start_time': time.time(),
            'end_time': None,
            'output_path': output_path
        }
        
        batch_jobs[job_id] = job
        
        # Start the processing thread if not running
        start_batch_thread()
        
        # Add job to the queue
        batch_queue.put(job_id)
        
        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "total_files": len(file_info_list),
            "output_path": output_path
        })
        
    except Exception as e:
        logger.error(f"Error in batch upscale endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-delete/<job_id>', methods=['POST'])
def batch_delete(job_id):
    try:
        if job_id not in batch_jobs:
            return jsonify({"error": "Batch job not found"}), 404
        
        job = batch_jobs[job_id]
        
        # If job is still processing, cancel it first
        if job['status'] == 'processing':
            job['cancelled'] = True
            job['status'] = 'cancelled'
        
        # Remove the job from the dictionary
        del batch_jobs[job_id]
        
        return jsonify({"status": "success", "message": "Job deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error in batch delete endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-status/<job_id>', methods=['GET'])
def batch_status(job_id):
    try:
        if job_id not in batch_jobs:
            return jsonify({"error": "Batch job not found"}), 404
        
        job = batch_jobs[job_id]
        
        # Create a simplified status response
        status_data = {
            'job_id': job['id'],
            'status': job['status'],
            'progress': job['progress'],
            'total_files': job['total_files'],
            'current_file': job['current_file'],
            'current_file_index': job.get('current_file_index'),
            'start_time': job['start_time'],
            'end_time': job['end_time'],
            'output_path': job.get('output_path', 'Default'),
            'files': [
                {
                    'filename': file_info['filename'],
                    'status': file_info['status'],
                    'error': file_info.get('error')
                }
                for file_info in job['files']
            ]
        }
        
        # Add download link if completed
        if job['status'] == 'completed' and 'zip_path' in job:
            status_data['download_url'] = url_for('batch_download', job_id=job_id, _external=True)
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Error in batch status endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-download/<job_id>', methods=['GET'])
def batch_download(job_id):
    try:
        if job_id not in batch_jobs:
            return jsonify({"error": "Batch job not found"}), 404
        
        job = batch_jobs[job_id]
        
        if job['status'] != 'completed' or 'zip_path' not in job:
            return jsonify({"error": "Batch job is not completed yet"}), 400
        
        # Send the zip file
        return send_file(
            job['zip_path'],
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"upscaled_batch_{job_id}.zip"
        )
        
    except Exception as e:
        logger.error(f"Error in batch download endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-cancel/<job_id>', methods=['POST'])
def batch_cancel(job_id):
    try:
        if job_id not in batch_jobs:
            return jsonify({"error": "Batch job not found"}), 404
        
        job = batch_jobs[job_id]
        
        if job['status'] in ['completed', 'failed']:
            return jsonify({"error": "Cannot cancel a completed or failed job"}), 400
        
        # Mark the job as cancelled
        job['cancelled'] = True
        job['status'] = 'cancelled'
        
        return jsonify({"status": "success", "message": "Job cancellation requested"})
        
    except Exception as e:
        logger.error(f"Error in batch cancel endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-list', methods=['GET'])
def batch_list():
    try:
        # Create a simplified list of all jobs
        jobs_list = [
            {
                'job_id': job['id'],
                'status': job['status'],
                'progress': job['progress'],
                'total_files': job['total_files'],
                'start_time': job['start_time'],
                'end_time': job['end_time'],
                'output_path': job.get('output_path', 'Default')
            }
            for job_id, job in batch_jobs.items()
        ]
        
        return jsonify(jobs_list)
        
    except Exception as e:
        logger.error(f"Error in batch list endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-cleanup', methods=['POST'])
def batch_cleanup():
    try:
        days = int(request.form.get('days', 7))
        
        # Calculate the cutoff time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0
        
        # Find jobs older than the cutoff
        jobs_to_remove = []
        for job_id, job in batch_jobs.items():
            if job['start_time'] < cutoff_time:
                jobs_to_remove.append(job_id)
        
        # Remove the jobs and their files
        for job_id in jobs_to_remove:
            job = batch_jobs[job_id]
            
            # Remove the job directory
            job_dir = os.path.join(batch_results_dir, job_id)
            if os.path.exists(job_dir):
                for file in os.listdir(job_dir):
                    os.remove(os.path.join(job_dir, file))
                os.rmdir(job_dir)
            
            # Remove the zip file if it exists
            if 'zip_path' in job and os.path.exists(job['zip_path']):
                os.remove(job['zip_path'])
            
            # Remove from the dictionary
            del batch_jobs[job_id]
            removed_count += 1
        
        return jsonify({
            "status": "success", 
            "message": f"Removed {removed_count} jobs older than {days} days"
        })
        
    except Exception as e:
        logger.error(f"Error in batch cleanup endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/resize', methods=['POST'])
def resize():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        scale_factor = float(request.form.get('scale_factor', 1.0))
        method = request.form.get('method', 'box')
        gamma_correction = config['Processing'].getboolean('GammaCorrection', False)
        
        # Validate scale factor
        if scale_factor < MIN_SCALE_FACTOR or scale_factor > MAX_SCALE_FACTOR:
            return jsonify({
                "error": f"Scale factor must be between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR} (inclusive)."
            }), 400
        
        # Validate resize method
        available_filters = get_available_filters()
        if method.lower() not in available_filters:
            return jsonify({
                "error": f"Unsupported method: {method}",
                "available_methods": available_filters
            }), 400
        
        image = Image.open(file.stream)
        logger.debug(f"Original image size: {image.size}")
        logger.debug(f"Original image mode: {image.mode}")
        logger.debug(f"Scale factor: {scale_factor}")
        logger.debug(f"Method: {method}")
        
        # Perform resizing using the full resize_image function from resize_module
        result = resize_image(
            image=image,
            scale_factor=scale_factor,
            method=method,
            gamma_correction=gamma_correction
        )
        
        logger.debug(f"Resized image size: {result.size}")
        logger.debug(f"Resized image mode: {result.mode}")
        
        # Save result
        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        # Create descriptive filename
        operation = "upscaled" if scale_factor > 1 else "downscaled"
        new_filename = f"{operation}_{scale_factor}x_{method}_{file.filename}"
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=new_filename
        )
        
    except ValueError as e:
        logger.error(f"Value error in resize endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in resize endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-resize', methods=['POST'])
def batch_resize():
    try:
        logger.debug("Starting batch resize request")
        
        if 'images[]' not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist('images[]')
        scale_factor = float(request.form.get('scale_factor', 1.0))
        method = request.form.get('method', 'box')
        output_path = request.form.get('output_path')
        gamma_correction = config['Processing'].getboolean('GammaCorrection', False)
        
        # Validate scale factor
        if scale_factor < MIN_SCALE_FACTOR or scale_factor > MAX_SCALE_FACTOR:
            return jsonify({
                "error": f"Scale factor must be between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR} (inclusive)."
            }), 400
        
        # Validate resize method
        available_filters = get_available_filters()
        if method.lower() not in available_filters:
            return jsonify({
                "error": f"Unsupported method: {method}",
                "available_methods": available_filters
            }), 400
        
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        # If no output path provided, use the default
        if not output_path:
            output_path = config['Paths'].get('DefaultOutputPath', 'output')
            
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
            
        logger.debug(f"Processing batch resize request: scale={scale_factor}, method={method}, files={len(files)}, output={output_path}")
        
        # Create a new batch job
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(batch_results_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Process all files immediately (resize is fast)
        operation = "upscaled" if scale_factor > 1 else "downscaled"
        
        # Create a zip file for the results
        zip_path = os.path.join(batch_results_dir, f"{job_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files:
                if file.filename:
                    try:
                        filename = secure_filename(file.filename)
                        new_filename = f"{operation}_{scale_factor}x_{method}_{filename}"
                        
                        # Process the image
                        image = Image.open(file.stream)
                        result = resize_image(
                            image=image,
                            scale_factor=scale_factor,
                            method=method,
                            gamma_correction=gamma_correction
                        )
                        
                        # Save to the job directory
                        job_output_path = os.path.join(job_dir, new_filename)
                        result.save(job_output_path, format='PNG')
                        
                        # Save to the user-specified output directory
                        user_output_path = os.path.join(output_path, new_filename)
                        result.save(user_output_path, format='PNG')
                        
                        # Add to the zip
                        zipf.write(job_output_path, arcname=new_filename)
                        
                    except Exception as e:
                        logger.error(f"Error processing {file.filename}: {str(e)}")
        
        # Send the zip file
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{operation}_batch_{scale_factor}x_{method}.zip"
        )
        
    except Exception as e:
        logger.error(f"Error in batch resize endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/resize/methods', methods=['GET'])
def get_resize_methods():
    """Get available resize methods and scale factor limits."""
    return jsonify({
        'methods': get_available_filters(),
        'scale_limits': {
            'min': MIN_SCALE_FACTOR,
            'max': MAX_SCALE_FACTOR
        }
    })

@app.route('/api/info', methods=['POST'])
def get_info():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        image = Image.open(file.stream)
        
        # Get image information
        info = get_image_metadata(image)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error in info endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Cleanup task
def cleanup_models_task():
    global models, last_cleanup_time
    while True:
        time.sleep(60)  # Check every minute
        current_time = time.time()
        if current_time - last_cleanup_time >= CLEANUP_INTERVAL:
            print("Performing periodic cleanup of unused models...")
            models.clear()
            torch.cuda.empty_cache()
            gc.collect()
            last_cleanup_time = current_time
            print("Cache cleanup completed. All models unloaded and memory freed.")

# Default configuration
DEFAULT_CONFIG = {
    'Paths': {
        'ModelPath': 'models',
        'DefaultOutputPath': 'output'  # Default output directory
    },
    'Processing': {
        'MaxTileSize': '1024',
        'Precision': 'auto',
        'MaxOutputTotalPixels': '67108864',
        'UpscaleTimeout': '60',
        'OtherStepTimeout': '30',
        'ThreadPoolWorkers': '1',
        'MaxConcurrentUpscales': '1',
        'DefaultAlphaHandling': 'resize',
        'GammaCorrection': 'false',
        'VRAMSafetyMultiplier': '1.2',
        'AvailableVRAMUsageFraction': '0.8',
        'DefaultTileSize': '384',
        'BatchJobRetentionDays': '7',
        'BatchConcurrentProcessing': 'true',  # Enable parallel processing by default
        'BatchParallelSize': '4',  # Default number of parallel batch tasks
        'PromptForSaveLocation': 'true'  # Ask for save location before processing
    }
}

# Initialize config with defaults if it doesn't exist
def init_config():
    if not os.path.exists('config.ini'):
        for section, values in DEFAULT_CONFIG.items():
            if section not in config:
                config.add_section(section)
            for key, value in values.items():
                config[section][key] = value
        
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    else:
        # Make sure the BatchParallelSize parameter exists
        if 'Processing' in config and 'BatchParallelSize' not in config['Processing']:
            config['Processing']['BatchParallelSize'] = DEFAULT_CONFIG['Processing']['BatchParallelSize']
            with open('config.ini', 'w') as configfile:
                config.write(configfile)

if __name__ == '__main__':
    # Initialize the config with defaults
    init_config()
    
    # Initialize the batch processing thread
    start_batch_thread()
    
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_models_task, daemon=True)
    cleanup_thread.start()
    
    app.run(host='localhost', port=5000, debug=True)
