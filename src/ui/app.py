"""
Main Gradio Application
Wakeword Training Platform with 6 panels
"""
import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.cuda_utils import enforce_cuda, get_cuda_validator
from src.config.logger import get_logger
from src.ui.panel_dataset import create_dataset_panel
from src.ui.panel_config import create_config_panel
from src.ui.panel_training import create_training_panel
from src.ui.panel_evaluation import create_evaluation_panel
from src.ui.panel_export import create_export_panel
from src.ui.panel_docs import create_docs_panel


def find_available_port(start_port: int = 7860, end_port: int = 7870) -> int:
    """
    Find an available port in the specified range

    Args:
        start_port: Starting port number
        end_port: Ending port number

    Returns:
        Available port number or start_port if none found
    """
    import socket

    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue

    # If no port available, return start_port and let it fail with clear message
    return start_port


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application with all panels

    Returns:
        Gradio Blocks app
    """
    # Validate CUDA
    logger = get_logger("app")
    logger.info("Starting Wakeword Training Platform")

    cuda_validator = enforce_cuda()
    logger.info("CUDA validation passed")

    # Create main app with theme
    with gr.Blocks(
        title="Wakeword Training Platform",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    ) as app:

        # Global state for sharing data between panels
        global_state = gr.State(value={'config': None})

        # Header
        gr.Markdown("""
        # 🎙️ Wakeword Training Platform
        ### GPU-Accelerated Custom Wakeword Detection Model Training

        Complete pipeline from dataset management to model deployment.
        """)

        # Display GPU info
        gpu_info = cuda_validator.get_device_info()
        gr.Markdown(f"""
        **GPU Status**: ✅ {gpu_info['device_count']} GPU(s) available |
        **CUDA Version**: {gpu_info['cuda_version']} |
        **Active Device**: {gpu_info['devices'][0]['name']} ({gpu_info['devices'][0]['total_memory_gb']} GB)
        """)

        gr.Markdown("---")

        # Create tabs for 6 panels
        with gr.Tabs():
            with gr.TabItem("📊 1. Dataset Management", id=1):
                panel_dataset = create_dataset_panel()

            with gr.TabItem("⚙️ 2. Configuration", id=2):
                panel_config = create_config_panel(global_state)

            with gr.TabItem("🚀 3. Training", id=3):
                panel_training = create_training_panel(global_state)

            with gr.TabItem("🎯 4. Evaluation", id=4):
                panel_evaluation = create_evaluation_panel()

            with gr.TabItem("📦 5. ONNX Export", id=5):
                panel_export = create_export_panel()

            with gr.TabItem("📚 6. Documentation", id=6):
                panel_docs = create_docs_panel()

        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        **Wakeword Training Platform v1.0** | Reliability-focused implementation |
        GPU-accelerated with PyTorch & CUDA
        """)

    logger.info("Application created successfully")
    return app


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = None,
    share: bool = False,
    inbrowser: bool = True
):
    """
    Launch the Gradio application

    Args:
        server_name: Server host
        server_port: Server port (None = auto-find between 7860-7870)
        share: Create public share link
        inbrowser: Open browser automatically
    """
    logger = get_logger("app")

    # Find available port if not specified
    if server_port is None:
        server_port = find_available_port(7860, 7870)
        logger.info(f"Auto-selected port: {server_port}")

    # Create and launch app
    app = create_app()

    logger.info(f"Launching app on {server_name}:{server_port}")
    logger.info(f"Share mode: {share}")
    logger.info(f"Open browser: {inbrowser}")

    try:
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            inbrowser=inbrowser,
            show_error=True,
            show_api=False,
            quiet=False
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.error(f"Port {server_port} is already in use")
            logger.info("Trying to find another available port...")
            new_port = find_available_port(server_port + 1, 7870)
            if new_port != server_port:
                logger.info(f"Retrying with port {new_port}")
                app.launch(
                    server_name=server_name,
                    server_port=new_port,
                    share=share,
                    inbrowser=inbrowser,
                    show_error=True,
                    quiet=False
                )
            else:
                logger.error("No available ports found in range 7860-7870")
                raise
        else:
            raise


if __name__ == "__main__":
    # Launch with default settings
    launch_app()