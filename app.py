import streamlit as st
import os
import json
import time
from datetime import datetime
from model_manager import ModelManager, LLAMA_CPP_AVAILABLE
from chat_interface import ChatInterface
from utils import format_file_size, export_conversation
from performance_monitor import PerformanceMonitor
from batch_processor import BatchProcessor

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

if 'chat_interface' not in st.session_state:
    st.session_state.chat_interface = ChatInterface()

if 'uploaded_models' not in st.session_state:
    st.session_state.uploaded_models = {}

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = "text"

if 'performance_monitor' not in st.session_state:
    st.session_state.performance_monitor = PerformanceMonitor()

if 'batch_processor' not in st.session_state:
    st.session_state.batch_processor = BatchProcessor()

if 'hardware_config' not in st.session_state:
    st.session_state.hardware_config = {
        'acceleration_mode': 'cpu_only',
        'n_gpu_layers': 0,
        'n_threads': os.cpu_count() or 4,
        'use_server_cpu': True
    }

def main():
    st.set_page_config(
        page_title="GGUF AI Model Runner",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 GGUF AI Model Runner")
    st.markdown("Upload and test various AI models from GGUF files locally")
    
    # Check dependency status
    if not LLAMA_CPP_AVAILABLE:
        st.error("⚠️ **llama-cpp-python is not installed.** This package is required to run GGUF models.")
        
        with st.expander("🔧 Installation Instructions & Troubleshooting", expanded=True):
            st.markdown("""
            ### Installation Methods
            
            **Method 1: Using Shell Terminal**
            1. Open the Shell tab in Replit (bottom panel or side panel)
            2. Install build dependencies first:
               ```bash
               # Install system dependencies (if needed)
               pip install cmake
               ```
            3. Install llama-cpp-python:
               ```bash
               # For CPU-only (recommended for Replit)
               CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_BLAS_VENDOR=Generic" pip install llama-cpp-python
               
               # Or try the simple version
               pip install llama-cpp-python --no-cache-dir
               ```
            
            **Method 2: Using Dependencies Tab**
            1. Click the "Dependencies" tab in the left sidebar
            2. Search for "llama-cpp-python"
            3. Click "Add" to install
            
            ### If Installation Fails
            
            **Common Issue:** Build errors with CMake
            - This package requires compilation which can fail in some environments
            - The build process needs CMake and C++ compiler tools
            
            **Alternative Solutions:**
            1. **Use Cloud AI APIs instead:** Consider using OpenAI, Anthropic, or Google's APIs for similar functionality
            2. **Try pre-built wheels:** Look for platform-specific pre-compiled versions
            3. **Use smaller models:** Some quantized models work better in resource-constrained environments
            
            **After Installation:**
            - Restart this application (refresh the page)
            - The green success message should appear above
            """)
            
        st.info("💡 **Alternative:** You can also use cloud-based AI APIs (OpenAI, Anthropic, etc.) which don't require local model compilation.")
    else:
        st.success("✅ All dependencies are installed! Ready to run GGUF models.")
    
    # Sidebar for model management
    with st.sidebar:
        st.header("Model Management")
        
        # Model upload section
        st.subheader("Upload GGUF Model")
        
        # Initialize uploaded_file variable
        uploaded_file = None
        
        # Add upload method selection
        upload_method = st.radio(
            "Choose upload method:",
            ["🌐 Web Upload (< 500MB)", "📁 Local File Path", "ℹ️ Large File Guide"],
            help="Select the best method for your file size"
        )
        
        if upload_method == "🌐 Web Upload (< 500MB)":
            st.info("📋 **Web Upload:**\n- Maximum: 500MB (recommended)\n- Direct upload through browser\n- Best for smaller GGUF models")
            
            uploaded_file = st.file_uploader(
                "Choose a GGUF file",
                type=['gguf'],
                help="Upload GGUF files smaller than 500MB for best stability"
            )
            
            # Show file size if file is selected
            if uploaded_file is not None:
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > 500:  # > 500MB
                    st.error(f"❌ File too large: {file_size_mb:.1f}MB. Please use 'Local File Path' method for files > 500MB.")
                    uploaded_file = None
                else:
                    st.success(f"✅ File size: {file_size_mb:.1f}MB - Ready for upload!")
        
        elif upload_method == "📁 Local File Path":
            st.info("📋 **Local File Path:**\n- For files already on the server\n- No size limitations\n- Paste the full path to your GGUF file")
            
            file_path_input = st.text_input(
                "Enter full path to GGUF file:",
                placeholder="/path/to/your/model.gguf",
                help="Enter the complete file path to a GGUF file on the server"
            )
            
            if file_path_input and st.button("📂 Load from Path"):
                if os.path.exists(file_path_input) and file_path_input.endswith('.gguf'):
                    # Create a mock uploaded file object
                    safe_filename = os.path.basename(file_path_input)
                    if safe_filename not in st.session_state.uploaded_models:
                        try:
                            with st.spinner(f"Loading {safe_filename} from local path..."):
                                model_info = st.session_state.model_manager.get_basic_model_info(file_path_input)
                                model_info['file_path'] = file_path_input
                                model_info['upload_time'] = datetime.now().isoformat()
                                model_info['original_name'] = safe_filename
                                model_info['source'] = 'local_path'
                                
                                st.session_state.uploaded_models[safe_filename] = model_info
                                st.success(f"✅ {safe_filename} loaded successfully from local path!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to load file: {str(e)}")
                    else:
                        st.warning("⚠️ This model is already loaded!")
                else:
                    st.error("❌ File not found or not a .gguf file. Please check the path.")
            
        
        else:  # Large File Guide
            st.info("📋 **For Large GGUF Files (> 500MB):**")
            
            with st.expander("🔧 Method 1: Upload via File Manager", expanded=True):
                st.markdown("""
                **Steps:**
                1. Open Replit's **Files** panel (left sidebar)
                2. Create a `models` folder if it doesn't exist
                3. Drag and drop your GGUF file into the `models` folder
                4. Come back here and use **"Local File Path"** method
                5. Enter path: `/home/runner/{YourReplName}/models/your-model.gguf`
                """)
            
            with st.expander("💡 Method 2: Use Smaller Models"):
                st.markdown("""
                **Recommended smaller GGUF models:**
                - **TinyLlama-1.1B** (~637MB) - Good for testing
                - **Phi-3-mini-4k** (~2.4GB) - Compact but powerful
                - **Qwen2-0.5B** (~374MB) - Very fast and small
                - **Gemma-2B** (~1.4GB) - Google's efficient model
                
                **Where to find:**
                - Hugging Face Hub (search for GGUF files)
                - Ollama model library
                - LlamaCpp model repositories
                """)
            
            with st.expander("⚡ Method 3: Model Quantization"):
                st.markdown("""
                **Reduce model size:**
                - Use Q4_0 or Q4_1 quantized versions (smaller files)
                - Q8_0 for better quality but larger size
                - Q2_K for extremely small files (lowest quality)
                
                **Tools for quantization:**
                - llama.cpp quantization tools
                - Ollama (automatically handles quantization)
                """)
        
        # Process uploaded file (only for web upload method)
        if upload_method == "🌐 Web Upload (< 500MB)" and uploaded_file is not None:
            # Sanitize filename to prevent path traversal attacks
            safe_filename = os.path.basename(uploaded_file.name)
            safe_filename = safe_filename.replace('/', '_').replace('\\', '_')
            
            if safe_filename not in st.session_state.uploaded_models:
                try:
                    with st.spinner(f"Processing {safe_filename}..."):
                        # Save uploaded file
                        os.makedirs("models", exist_ok=True)
                        file_path = os.path.join("models", safe_filename)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Get basic model info (avoid loading full model during upload)
                        model_info = st.session_state.model_manager.get_basic_model_info(file_path)
                        model_info['file_path'] = file_path
                        model_info['upload_time'] = datetime.now().isoformat()
                        model_info['original_name'] = uploaded_file.name
                        model_info['source'] = 'web_upload'
                        
                        st.session_state.uploaded_models[safe_filename] = model_info
                        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    if "502" in str(e) or "timeout" in str(e).lower():
                        st.error("🔄 **Large file upload tips:**\n- Try uploading during off-peak hours\n- Ensure stable internet connection\n- Consider using smaller model files (< 500MB)")
                    st.info("💡 **Alternative approaches:**\n- Use quantized models (smaller file sizes)\n- Try uploading in multiple attempts\n- Check your internet connection stability")
        
        # Model selection
        st.subheader("Available Models")
        if st.session_state.uploaded_models:
            model_names = list(st.session_state.uploaded_models.keys())
            selected_model = st.selectbox(
                "Select a model to use:",
                options=[None] + model_names,
                format_func=lambda x: "Select a model..." if x is None else st.session_state.uploaded_models[x].get('original_name', x)
            )
            
            if selected_model and selected_model != st.session_state.current_model:
                with st.spinner(f"Loading {selected_model}..."):
                    success = st.session_state.model_manager.load_model(
                        st.session_state.uploaded_models[selected_model]['file_path'],
                        hardware_config=st.session_state.hardware_config
                    )
                    if success:
                        st.session_state.current_model = selected_model
                        st.success(f"✅ {selected_model} loaded!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to load {selected_model}")
            
            # Display current model info
            if st.session_state.current_model:
                st.subheader("Current Model Info")
                model_info = st.session_state.uploaded_models[st.session_state.current_model]
                
                original_name = model_info.get('original_name', st.session_state.current_model)
                st.write(f"**Name:** {original_name}")
                st.write(f"**Size:** {format_file_size(model_info.get('file_size', 0))}")
                st.write(f"**Parameters:** {model_info.get('parameters', 'Unknown')}")
                st.write(f"**Type:** {model_info.get('model_type', 'Unknown')}")
                st.write(f"**Architecture:** {model_info.get('architecture', 'Unknown')}")
                st.write(f"**Uploaded:** {model_info.get('upload_time', 'Unknown')}")
                
                # Add button to get detailed metadata if llama-cpp is available
                if LLAMA_CPP_AVAILABLE and st.button("🔍 Get Detailed Metadata"):
                    with st.spinner("Loading detailed model information..."):
                        detailed_info = st.session_state.model_manager.get_model_info(model_info['file_path'])
                        st.session_state.uploaded_models[st.session_state.current_model].update(detailed_info)
                        st.rerun()
                
                # Performance monitoring section
                st.subheader("📊 Performance Monitor")
                perf_summary = st.session_state.performance_monitor.get_performance_summary()
                
                if "status" in perf_summary:
                    st.info(perf_summary["status"])
                else:
                    # Display performance metrics in columns
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        st.metric("Total Generations", perf_summary.get("Total Generations", "0"))
                        st.metric("Average Speed", perf_summary.get("Average Speed", "0 tokens/sec"))
                        st.metric("Current Memory", perf_summary.get("Current Memory Usage", "0 MB"))
                    
                    with perf_col2:
                        st.metric("Average Response Time", perf_summary.get("Average Response Time", "0s"))
                        st.metric("Total Tokens", perf_summary.get("Total Tokens Generated", "0"))
                        st.metric("Current CPU", perf_summary.get("Current CPU Usage", "0%"))
                    
                    # Performance actions
                    perf_action_col1, perf_action_col2 = st.columns(2)
                    with perf_action_col1:
                        if st.button("📈 View Detailed Metrics"):
                            st.session_state.show_detailed_metrics = True
                            st.rerun()
                    
                    with perf_action_col2:
                        if st.button("🗑️ Clear Performance Data"):
                            st.session_state.performance_monitor.clear_metrics()
                            st.rerun()
        else:
            st.info("No models uploaded yet. Upload a GGUF file to get started.")
        
        # Hardware Configuration Section
        st.subheader("⚡ Hardware Acceleration")
        
        # Hardware acceleration mode selection
        acceleration_modes = {
            'cpu_only': '🖥️ CPU Only (Default)',
            'gpu_partial': '🎮 GPU Partial (Some layers)',
            'gpu_full': '🚀 GPU Full (All layers)',
            'cpu_gpu_hybrid': '⚖️ CPU + GPU Hybrid',
            'server_cpu': '☁️ Server CPU Optimized'
        }
        
        selected_mode = st.selectbox(
            "Acceleration Mode:",
            options=list(acceleration_modes.keys()),
            format_func=lambda x: acceleration_modes[x],
            index=list(acceleration_modes.keys()).index(st.session_state.hardware_config['acceleration_mode']),
            help="Choose how to utilize hardware for model inference"
        )
        
        # Update hardware config when mode changes
        if selected_mode != st.session_state.hardware_config['acceleration_mode']:
            st.session_state.hardware_config['acceleration_mode'] = selected_mode
            
        # Apply button for hardware changes
        if st.session_state.current_model:
            if st.button("🔄 Apply Hardware Settings", help="Reload current model with new hardware settings"):
                with st.spinner("Reloading model with new hardware settings..."):
                    current_model_path = st.session_state.uploaded_models[st.session_state.current_model]['file_path']
                    success = st.session_state.model_manager.load_model(
                        current_model_path,
                        hardware_config=st.session_state.hardware_config
                    )
                    if success:
                        st.success("✅ Hardware settings applied successfully!")
                    else:
                        st.error("❌ Failed to apply hardware settings")
                    st.rerun()
            
        # Show mode-specific options
        if selected_mode == 'cpu_only':
            st.info("🖥️ **CPU Only Mode**\nUses only CPU cores for inference. Most compatible but slower.")
            threads = st.slider(
                "CPU Threads", 
                min_value=1, max_value=min(32, (os.cpu_count() or 4) * 2), 
                value=st.session_state.hardware_config['n_threads'],
                help="Number of CPU threads to use"
            )
            st.session_state.hardware_config['n_threads'] = threads
            st.session_state.hardware_config['n_gpu_layers'] = 0
            
        elif selected_mode == 'gpu_partial':
            st.info("🎮 **GPU Partial Mode**\nOffloads some model layers to GPU while keeping others on CPU.")
            gpu_layers = st.slider(
                "GPU Layers", 
                min_value=1, max_value=50, 
                value=max(1, st.session_state.hardware_config['n_gpu_layers']),
                help="Number of model layers to run on GPU"
            )
            st.session_state.hardware_config['n_gpu_layers'] = gpu_layers
            st.warning("⚠️ Requires CUDA-compatible GPU and proper drivers")
            
        elif selected_mode == 'gpu_full':
            st.info("🚀 **GPU Full Mode**\nRuns all possible layers on GPU for maximum speed.")
            st.session_state.hardware_config['n_gpu_layers'] = -1  # All layers
            st.warning("⚠️ Requires powerful GPU with sufficient VRAM")
            
        elif selected_mode == 'cpu_gpu_hybrid':
            st.info("⚖️ **CPU + GPU Hybrid**\nBalances workload between CPU and GPU.")
            col1, col2 = st.columns(2)
            with col1:
                gpu_layers = st.slider(
                    "GPU Layers", 
                    min_value=0, max_value=30, 
                    value=st.session_state.hardware_config['n_gpu_layers'],
                    help="Layers on GPU"
                )
            with col2:
                threads = st.slider(
                    "CPU Threads", 
                    min_value=1, max_value=min(16, os.cpu_count() or 4), 
                    value=st.session_state.hardware_config['n_threads'],
                    help="CPU threads"
                )
            st.session_state.hardware_config['n_gpu_layers'] = gpu_layers
            st.session_state.hardware_config['n_threads'] = threads
            
        elif selected_mode == 'server_cpu':
            st.info("☁️ **Server CPU Optimized**\nOptimized for server/cloud environments with limited resources.")
            threads = st.slider(
                "CPU Threads", 
                min_value=1, max_value=8, 
                value=min(4, st.session_state.hardware_config['n_threads']),
                help="Conservative CPU usage for shared environments"
            )
            st.session_state.hardware_config['n_threads'] = threads
            st.session_state.hardware_config['n_gpu_layers'] = 0
            st.session_state.hardware_config['use_server_cpu'] = True
        
        # Hardware detection info
        with st.expander("🔍 Hardware Detection & Capabilities"):
            import psutil
            
            # CPU Information
            st.write(f"**CPU Cores:** {os.cpu_count()} cores available")
            st.write(f"**Memory:** {psutil.virtual_memory().total / (1024**3):.1f} GB total")
            
            # Check llama-cpp-python build capabilities
            gpu_available = False
            cuda_available = False
            
            if LLAMA_CPP_AVAILABLE:
                try:
                    # Check if llama-cpp-python was built with CUDA support
                    import llama_cpp
                    if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp._lib, 'llama_supports_gpu_offload'):
                        gpu_available = llama_cpp._lib.llama_supports_gpu_offload()
                        st.write(f"**llama-cpp GPU Support:** {'✅ Available' if gpu_available else '❌ Not compiled with GPU support'}")
                    else:
                        st.write("**llama-cpp GPU Support:** ❓ Unable to detect")
                        
                except Exception as e:
                    st.write(f"**llama-cpp GPU Support:** ❓ Error checking: {str(e)}")
            else:
                st.write("**llama-cpp GPU Support:** ❌ llama-cpp-python not installed")
            
            # NVIDIA GPU Detection
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    st.write("**NVIDIA GPU Detected:**")
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 3:
                                name, memory, driver = parts[0], parts[1], parts[2]
                                st.write(f"  - {name.strip()} ({memory.strip()} MB VRAM, Driver: {driver.strip()})")
                                cuda_available = True
                else:
                    st.write("**NVIDIA GPU:** ❌ No NVIDIA GPU detected")
            except FileNotFoundError:
                st.write("**NVIDIA GPU:** ❌ nvidia-smi not found (no NVIDIA drivers)")
            except Exception as e:
                st.write(f"**NVIDIA GPU:** ❓ Detection failed: {str(e)}")
                
            # Recommendations based on detection
            st.subheader("💡 Recommendations")
            if not LLAMA_CPP_AVAILABLE:
                st.warning("Install llama-cpp-python first to use any acceleration modes")
            elif gpu_available and cuda_available:
                st.success("🚀 GPU acceleration fully supported! You can use all acceleration modes.")
            elif cuda_available and not gpu_available:
                st.warning("🔧 NVIDIA GPU detected but llama-cpp-python not compiled with CUDA. Reinstall with CUDA support for GPU acceleration.")
            elif gpu_available and not cuda_available:
                st.info("⚡ llama-cpp-python has GPU support but no NVIDIA GPU detected. CPU modes recommended.")
            else:
                st.info("🖥️ CPU-only environment detected. Use 'CPU Only' or 'Server CPU Optimized' modes.")
        
        # Model management actions
        if st.session_state.uploaded_models:
            st.subheader("Model Actions")
            if st.button("🗑️ Clear All Models"):
                # Unload current model
                st.session_state.model_manager.unload_model()
                
                # Remove model files
                for model_info in st.session_state.uploaded_models.values():
                    if os.path.exists(model_info['file_path']):
                        os.remove(model_info['file_path'])
                
                # Clear session state
                st.session_state.uploaded_models = {}
                st.session_state.current_model = None
                st.session_state.conversation_history = []
                
                st.success("All models cleared!")
                st.rerun()
    
    # Main content area
    if st.session_state.current_model:
        # Chat interface
        st.header(f"💬 Chat with {st.session_state.current_model}")
        
        # Model type selection
        col1, col2 = st.columns([1, 2])
        with col1:
            model_types = st.session_state.model_manager.get_supported_model_types()
            selected_type = st.selectbox(
                "Model Type:",
                options=model_types,
                index=model_types.index(st.session_state.selected_model_type),
                help="Choose the type of AI model interaction"
            )
            
            if selected_type != st.session_state.selected_model_type:
                st.session_state.selected_model_type = selected_type
                st.session_state.model_manager.set_model_type(selected_type)
                st.rerun()
        
        with col2:
            type_descriptions = {
                "text": "📝 General conversation and text generation",
                "code": "💻 Programming assistance and code generation", 
                "multimodal": "🎭 Multi-format content understanding"
            }
            st.info(type_descriptions.get(selected_type, "General AI interaction"))
        
        # Advanced parameters in expandable section
        with st.expander("🎛️ Advanced Generation Parameters", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, max_value=2.0, value=0.7, step=0.05,
                    help="Controls randomness. Lower = more focused, Higher = more creative"
                )
                top_p = st.slider(
                    "Top P (Nucleus)", 
                    min_value=0.0, max_value=1.0, value=0.9, step=0.05,
                    help="Considers only tokens with cumulative probability up to this value"
                )
                
            with col2:
                max_tokens = st.slider(
                    "Max Tokens", 
                    min_value=50, max_value=4096, value=512, step=50,
                    help="Maximum number of tokens to generate"
                )
                repeat_penalty = st.slider(
                    "Repeat Penalty", 
                    min_value=0.9, max_value=1.3, value=1.1, step=0.05,
                    help="Penalty for repeating tokens. Higher = less repetition"
                )
                
            with col3:
                top_k = st.slider(
                    "Top K", 
                    min_value=1, max_value=100, value=40, step=5,
                    help="Consider only the K most likely tokens"
                )
                seed = st.number_input(
                    "Seed (optional)", 
                    min_value=-1, max_value=2147483647, value=-1,
                    help="Set to -1 for random seed, or provide specific seed for reproducible results"
                )
            
            # Preset configurations
            st.subheader("Parameter Presets")
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            
            with preset_col1:
                if st.button("🎯 Focused (Low Temperature)"):
                    st.session_state.preset_params = {
                        'temperature': 0.2, 'top_p': 0.8, 'top_k': 20, 
                        'repeat_penalty': 1.15, 'max_tokens': 512
                    }
                    st.rerun()
                    
            with preset_col2:
                if st.button("⚖️ Balanced (Default)"):
                    st.session_state.preset_params = {
                        'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 
                        'repeat_penalty': 1.1, 'max_tokens': 512
                    }
                    st.rerun()
                    
            with preset_col3:
                if st.button("🎨 Creative (High Temperature)"):
                    st.session_state.preset_params = {
                        'temperature': 1.2, 'top_p': 0.95, 'top_k': 60, 
                        'repeat_penalty': 1.05, 'max_tokens': 768
                    }
                    st.rerun()
            
            # Apply preset if available
            if hasattr(st.session_state, 'preset_params'):
                params = st.session_state.preset_params
                temperature = params['temperature']
                top_p = params['top_p'] 
                top_k = params['top_k']
                repeat_penalty = params['repeat_penalty']
                max_tokens = params['max_tokens']
                delattr(st.session_state, 'preset_params')  # Clear after use
        
        # Chat messages display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.conversation_history:
                if message['role'] == 'user':
                    st.chat_message("user").write(message['content'])
                else:
                    st.chat_message("assistant").write(message['content'])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Display user message
            with chat_container:
                st.chat_message("user").write(user_input)
            
            # Generate response with advanced parameters
            generation_params = {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'model_type': st.session_state.selected_model_type
            }
            
            # Add advanced parameters if they exist
            if 'repeat_penalty' in locals():
                generation_params['repeat_penalty'] = repeat_penalty
            if 'top_k' in locals():
                generation_params['top_k'] = top_k  
            if 'seed' in locals() and seed != -1:
                generation_params['seed'] = seed
                
            # Start performance monitoring
            st.session_state.performance_monitor.start_generation_timing()
            
            with st.spinner("Generating response..."):
                response_data = st.session_state.model_manager.generate_response(
                    user_input,
                    **generation_params
                )
                
                response = response_data.get('text')
                usage = response_data.get('usage', {})
                
                # End performance monitoring with proper metrics
                metrics = st.session_state.performance_monitor.end_generation_timing(
                    response=response or "",
                    prompt=user_input,
                    response_metadata=usage
                )
                
                # Store latest metrics for immediate display
                if metrics:
                    st.session_state.latest_generation_metrics = metrics
                
                if response:
                    # Add assistant response to history
                    st.session_state.conversation_history.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Display assistant response
                    with chat_container:
                        st.chat_message("assistant").write(response)
                    
                    st.rerun()
                else:
                    st.error("Failed to generate response. Please try again.")
        
        # Conversation management
        if st.session_state.conversation_history:
            st.subheader("Conversation Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Conversation"):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            with col2:
                if st.button("💾 Export Conversation"):
                    export_data = export_conversation(
                        st.session_state.conversation_history,
                        st.session_state.current_model
                    )
                    st.download_button(
                        label="Download JSON",
                        data=export_data,
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # Detailed metrics view (if requested)
        if hasattr(st.session_state, 'show_detailed_metrics') and st.session_state.show_detailed_metrics:
            st.header("📈 Detailed Performance Metrics")
            
            # Export performance data
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Export Performance Data"):
                    perf_data = st.session_state.performance_monitor.export_metrics()
                    st.download_button(
                        label="Download Performance JSON",
                        data=perf_data,
                        file_name=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🔄 Refresh Metrics"):
                    st.rerun()
            
            with col3:
                if st.button("❌ Close Detailed View"):
                    st.session_state.show_detailed_metrics = False
                    st.rerun()
            
            # Recent metrics table
            recent_metrics = st.session_state.performance_monitor.get_recent_metrics(20)
            if recent_metrics:
                st.subheader("Recent Generations")
                
                # Create a simplified metrics table
                metrics_data = []
                for i, metric in enumerate(recent_metrics):
                    metrics_data.append({
                        "#": len(recent_metrics) - i,
                        "Time": metric['timestamp'].split('T')[1][:8] if 'T' in metric['timestamp'] else metric['timestamp'][:8],
                        "Response Time (s)": f"{metric['generation_time']:.2f}",
                        "Tokens/sec": f"{metric['tokens_per_second']:.1f}",
                        "Response Tokens": metric['response_tokens'],
                        "Memory Delta (MB)": f"{metric['memory_delta']:.1f}",
                        "Temperature": metric['generation_params'].get('temperature', 'N/A')
                    })
                
                st.dataframe(metrics_data, use_container_width=True)
                
                # Performance charts
                if len(recent_metrics) > 1:
                    st.subheader("Performance Trends")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Response time chart
                        response_times = [m['generation_time'] for m in recent_metrics]
                        st.line_chart(
                            response_times,
                            height=200
                        )
                        st.caption("Response Time Trend")
                    
                    with chart_col2:
                        # Tokens per second chart
                        tokens_per_sec = [m['tokens_per_second'] for m in recent_metrics]
                        st.line_chart(
                            tokens_per_sec,
                            height=200
                        )
                        st.caption("Tokens/Second Trend")
            else:
                st.info("No detailed metrics available yet. Start generating responses to see performance data.")
    else:
        # Welcome screen with batch processing option
        st.header("Welcome to GGUF AI Model Runner")
        
        # Main tabs for different modes  
        tab1, tab2, tab3 = st.tabs(["💬 Interactive Chat", "📄 Batch Processing", "⚔️ Model Comparison"])
        
        with tab1:
            st.markdown("""
            This application allows you to:
            
            1. **Upload GGUF Models**: Upload your GGUF format AI model files
            2. **Load and Switch Models**: Easily switch between different uploaded models
            3. **Interactive Chat**: Test your models with an interactive chat interface
            4. **Model Information**: View detailed information about your models
            5. **Conversation History**: Keep track of your conversations and export them
            
            ### Getting Started
            1. Upload a GGUF model file using the sidebar
            2. Select the uploaded model to load it
            3. Start chatting with your AI model!
            
            ### Supported Features
            - Multiple model support with different types (text, code, multimodal)
            - Real-time model switching
            - Advanced generation parameters with presets
            - Performance monitoring and benchmarks
            - Conversation export functionality
            - Model metadata display
            """)
            
            st.info("👈 Upload a GGUF model file from the sidebar to get started!")
        
        with tab2:
            st.markdown("""
            ### Batch Processing Mode
            
            Process multiple inputs at once for efficient bulk operations:
            
            - **Text Input**: Paste multiple prompts separated by new lines
            - **File Upload**: Upload CSV, JSON, or text files with multiple inputs
            - **Progress Tracking**: Monitor processing progress in real-time
            - **Export Results**: Download results in multiple formats (JSON, CSV, TXT)
            - **Error Handling**: Retry failed items individually
            
            Perfect for:
            - Content generation at scale
            - Testing model responses across different inputs
            - Automated content processing workflows
            """)
            
            if st.session_state.uploaded_models:
                # Batch processing interface
                st.subheader("🔄 Create New Batch")
                
                batch_input_method = st.radio(
                    "Input Method:",
                    ["Text Input", "File Upload"],
                    horizontal=True
                )
                
                if batch_input_method == "Text Input":
                    batch_text = st.text_area(
                        "Enter multiple inputs (one per line):",
                        height=150,
                        placeholder="Input 1\nInput 2\nInput 3\n..."
                    )
                    
                    if st.button("📝 Create Batch from Text"):
                        if batch_text.strip():
                            batch_id = st.session_state.batch_processor.create_batch_from_text(batch_text)
                            st.success(f"✅ Created batch {batch_id} with {len(st.session_state.batch_processor.batch_items)} items")
                            st.rerun()
                        else:
                            st.error("Please enter some text inputs")
                
                else:  # File Upload
                    batch_file = st.file_uploader(
                        "Upload batch file (CSV, JSON, or TXT):",
                        type=['csv', 'json', 'txt'],
                        help="CSV: First column as input. JSON: Array of strings or objects with 'text' field. TXT: One input per line."
                    )
                    
                    if batch_file and st.button("📁 Create Batch from File"):
                        try:
                            batch_id = st.session_state.batch_processor.create_batch_from_file(batch_file)
                            st.success(f"✅ Created batch {batch_id} with {len(st.session_state.batch_processor.batch_items)} items")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating batch: {e}")
                
                # Display current batch status
                batch_status = st.session_state.batch_processor.get_batch_status()
                if batch_status["status"] != "no_batch":
                    st.subheader("📊 Current Batch Status")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Items", batch_status["total"])
                    with col2:
                        st.metric("Completed", batch_status["completed"])
                    with col3:
                        st.metric("Failed", batch_status["failed"])
                    with col4:
                        st.metric("Progress", f"{batch_status['progress']:.1%}")
                    
                    # Progress bar
                    st.progress(batch_status["progress"])
                    
                    # Processing controls
                    control_col1, control_col2, control_col3 = st.columns(3)
                    
                    with control_col1:
                        if not batch_status["is_processing"] and batch_status["pending"] > 0:
                            if st.button("▶️ Start Processing"):
                                # Use current generation parameters
                                generation_params = {
                                    'temperature': 0.7,
                                    'max_tokens': 512,
                                    'top_p': 0.9,
                                    'model_type': st.session_state.selected_model_type
                                }
                                st.session_state.batch_processor.process_batch(
                                    st.session_state.model_manager,
                                    generation_params
                                )
                                st.rerun()
                    
                    with control_col2:
                        if batch_status["failed"] > 0:
                            if st.button("🔄 Retry Failed"):
                                generation_params = {
                                    'temperature': 0.7,
                                    'max_tokens': 512,
                                    'top_p': 0.9,
                                    'model_type': st.session_state.selected_model_type
                                }
                                st.session_state.batch_processor.retry_failed_items(
                                    st.session_state.model_manager,
                                    generation_params
                                )
                                st.rerun()
                    
                    with control_col3:
                        if st.button("🗑️ Clear Batch"):
                            st.session_state.batch_processor.clear_batch()
                            st.rerun()
                    
                    # Export results if there are completed items
                    if batch_status["completed"] > 0:
                        st.subheader("📥 Export Results")
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            if st.button("📄 Export as JSON"):
                                json_data = st.session_state.batch_processor.export_results("json")
                                st.download_button(
                                    "Download JSON",
                                    json_data,
                                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    "application/json"
                                )
                        
                        with export_col2:
                            if st.button("📊 Export as CSV"):
                                csv_data = st.session_state.batch_processor.export_results("csv")
                                st.download_button(
                                    "Download CSV",
                                    csv_data,
                                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv"
                                )
                        
                        with export_col3:
                            if st.button("📝 Export as TXT"):
                                txt_data = st.session_state.batch_processor.export_results("txt")
                                st.download_button(
                                    "Download TXT",
                                    txt_data,
                                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    "text/plain"
                                )
                        
                        # Show results preview
                        with st.expander("👁️ Preview Results", expanded=False):
                            results = st.session_state.batch_processor.get_results()
                            for result in results[-5:]:  # Show last 5 results
                                with st.container():
                                    st.write(f"**{result['id']}** - {result['status']}")
                                    st.write(f"Input: {result['input'][:100]}...")
                                    if result['response']:
                                        st.write(f"Response: {result['response'][:200]}...")
                                    if result['error']:
                                        st.error(f"Error: {result['error']}")
                                    st.write("---")
            
            else:
                st.info("👈 Upload and load a GGUF model first to use batch processing")
        
        with tab3:
            st.markdown("""
            ### Model Comparison Mode
            
            Compare responses from different models side-by-side:
            
            - **Multi-Model Selection**: Choose 2-4 models for comparison
            - **Synchronized Input**: Send the same prompt to all selected models
            - **Side-by-Side Results**: Compare responses in a clean layout
            - **Performance Comparison**: See response times and quality differences
            - **Export Comparisons**: Save comparison results for analysis
            
            Perfect for:
            - Model evaluation and selection
            - A/B testing different models
            - Understanding model strengths and weaknesses
            - Research and benchmarking
            """)
            
            if len(st.session_state.uploaded_models) >= 2:
                # Model selection for comparison
                st.subheader("🎯 Select Models for Comparison")
                
                available_models = list(st.session_state.uploaded_models.keys())
                selected_models = st.multiselect(
                    "Choose 2-4 models to compare:",
                    available_models,
                    default=available_models[:min(2, len(available_models))],
                    max_selections=4
                )
                
                if len(selected_models) >= 2:
                    # Comparison input
                    st.subheader("💭 Input for Comparison")
                    comparison_input = st.text_area(
                        "Enter your prompt:",
                        height=100,
                        placeholder="Enter a prompt to test with all selected models..."
                    )
                    
                    # Model type selection for comparison
                    model_type_comparison = st.selectbox(
                        "Model Type for Comparison:",
                        options=["text", "code", "multimodal"],
                        index=0
                    )
                    
                    # Advanced parameters for comparison
                    with st.expander("⚙️ Comparison Parameters"):
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            comp_temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                            comp_max_tokens = st.slider("Max Tokens", 50, 2048, 512)
                        with comp_col2:
                            comp_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
                            comp_repeat_penalty = st.slider("Repeat Penalty", 0.9, 1.3, 1.1, 0.05)
                    
                    # Run comparison
                    if st.button("🚀 Run Comparison", type="primary"):
                        if comparison_input.strip():
                            # Initialize comparison results if not exists
                            if 'comparison_results' not in st.session_state:
                                st.session_state.comparison_results = []
                            
                            comparison_params = {
                                'temperature': comp_temperature,
                                'max_tokens': comp_max_tokens,
                                'top_p': comp_top_p,
                                'repeat_penalty': comp_repeat_penalty,
                                'model_type': model_type_comparison
                            }
                            
                            # Create a new comparison
                            comparison = {
                                'id': f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                'timestamp': datetime.now().isoformat(),
                                'input': comparison_input,
                                'model_type': model_type_comparison,
                                'parameters': comparison_params,
                                'results': {}
                            }
                            
                            # Generate responses from each model
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, model_name in enumerate(selected_models):
                                status_text.text(f"Generating response from {model_name}...")
                                
                                # Load model if not current
                                if st.session_state.current_model != model_name:
                                    model_path = st.session_state.uploaded_models[model_name]['file_path']
                                    success = st.session_state.model_manager.load_model(model_path, hardware_config=st.session_state.hardware_config)
                                    if success:
                                        st.session_state.current_model = model_name
                                
                                # Generate response with timing
                                start_time = time.time()
                                response_data = st.session_state.model_manager.generate_response(
                                    comparison_input,
                                    **comparison_params
                                )
                                response = response_data.get('text')
                                usage = response_data.get('usage', {})
                                generation_time = time.time() - start_time
                                
                                # Store result
                                comparison['results'][model_name] = {
                                    'response': response or "No response generated",
                                    'generation_time': generation_time,
                                    'model_info': st.session_state.uploaded_models[model_name],
                                    'usage': usage
                                }
                                
                                progress_bar.progress((i + 1) / len(selected_models))
                            
                            status_text.text("Comparison complete!")
                            time.sleep(1)
                            status_text.empty()
                            progress_bar.empty()
                            
                            # Store comparison
                            st.session_state.comparison_results.append(comparison)
                            st.success("✅ Comparison completed!")
                            st.rerun()
                        
                        else:
                            st.error("Please enter a prompt for comparison")
                    
                    # Display comparison results
                    if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results:
                        st.subheader("📊 Comparison Results")
                        
                        # Select which comparison to view
                        comparison_options = [
                            f"{comp['id']} - {comp['input'][:50]}..." 
                            for comp in st.session_state.comparison_results
                        ]
                        
                        selected_comparison_idx = st.selectbox(
                            "Select comparison to view:",
                            range(len(comparison_options)),
                            format_func=lambda x: comparison_options[x],
                            index=len(comparison_options) - 1  # Default to latest
                        )
                        
                        comparison = st.session_state.comparison_results[selected_comparison_idx]
                        
                        # Display input and parameters
                        st.write(f"**Input:** {comparison['input']}")
                        st.write(f"**Model Type:** {comparison['model_type']}")
                        st.write(f"**Timestamp:** {comparison['timestamp']}")
                        
                        # Display results in columns
                        num_models = len(comparison['results'])
                        cols = st.columns(num_models)
                        
                        for i, (model_name, result) in enumerate(comparison['results'].items()):
                            with cols[i]:
                                st.subheader(f"🤖 {model_name}")
                                
                                # Model info
                                model_info = result['model_info']
                                st.caption(f"Size: {format_file_size(model_info.get('file_size', 0))}")
                                st.caption(f"Type: {model_info.get('model_type', 'Unknown')}")
                                
                                # Performance metrics
                                st.metric(
                                    "Response Time", 
                                    f"{result['generation_time']:.2f}s"
                                )
                                
                                # Response
                                st.write("**Response:**")
                                with st.container():
                                    st.write(result['response'])
                                
                                # Quality indicators (simple metrics)
                                response_length = len(result['response']) if result['response'] else 0
                                st.caption(f"Response length: {response_length} characters")
                        
                        # Comparison summary
                        st.subheader("📈 Comparison Summary")
                        
                        summary_data = []
                        for model_name, result in comparison['results'].items():
                            summary_data.append({
                                "Model": model_name,
                                "Response Time (s)": f"{result['generation_time']:.2f}",
                                "Response Length": len(result['response']) if result['response'] else 0,
                                "Characters/Second": f"{len(result['response']) / result['generation_time']:.1f}" if result['generation_time'] > 0 and result['response'] else "0"
                            })
                        
                        st.dataframe(summary_data, use_container_width=True)
                        
                        # Export comparison
                        if st.button("📥 Export Comparison"):
                            export_data = {
                                'comparison_id': comparison['id'],
                                'timestamp': comparison['timestamp'],
                                'input': comparison['input'],
                                'model_type': comparison['model_type'],
                                'parameters': comparison['parameters'],
                                'results': comparison['results'],
                                'summary': summary_data
                            }
                            
                            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                "Download Comparison JSON",
                                json_data,
                                f"model_comparison_{comparison['id']}.json",
                                "application/json"
                            )
                        
                        # Clear comparisons
                        if st.button("🗑️ Clear All Comparisons"):
                            st.session_state.comparison_results = []
                            st.rerun()
                
                else:
                    st.warning("Please select at least 2 models for comparison")
            
            elif len(st.session_state.uploaded_models) == 1:
                st.info("You need at least 2 models to use comparison mode. Upload more GGUF models.")
            
            else:
                st.info("👈 Upload at least 2 GGUF models to use comparison mode")

if __name__ == "__main__":
    main()
