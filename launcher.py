# launcher.py
# Este es el punto de entrada del ejecutable .exe
# Abre la app Streamlit en el navegador del usuario de forma automática.

import sys
import os
import threading
import webbrowser
import time
import socket

def find_free_port(start=8501):
    """Encuentra un puerto libre para Streamlit."""
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start

def get_base_dir():
    """Devuelve la ruta base correcta tanto en desarrollo como en .exe"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def open_browser(port, delay=3.5):
    """Abre el navegador después de un breve retraso."""
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")

def run_streamlit():
    base_dir  = get_base_dir()
    app_path  = os.path.join(base_dir, "main_app.py")
    port      = find_free_port()

    # Abrir el navegador en paralelo
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Variables de entorno para que Streamlit funcione en modo embedded
    os.environ["STREAMLIT_SERVER_PORT"]            = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"]         = "localhost"
    os.environ["STREAMLIT_SERVER_HEADLESS"]        = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_THEME_BASE"]             = "light"
    os.environ["STREAMLIT_THEME_PRIMARY_COLOR"]    = "#006847"

    # Ruta de datos incrustados
    os.environ["DATA_FLOW_BASE_DIR"] = base_dir

    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#006847",
        "--theme.backgroundColor", "#f4f6f5",
        "--theme.secondaryBackgroundColor", "#e8f5ee",
    ]
    stcli.main()

if __name__ == "__main__":
    run_streamlit()
