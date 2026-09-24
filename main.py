# FreshChain Application Entrypoint
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import app

if __name__ == "__main__":
    print("=" * 60)
    print("FreshChain AI - Cold Chain Spoilage Monitoring System")
    print("Server running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
