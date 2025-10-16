# Learned-Index-Structures
Prototype of learned index structures in Python, inspired by The Case for Learned Index Structures. Implements recursive model indexes (RMIs) and compares them against traditional B-Trees, with benchmarks on lookup speed, memory use, and accuracy.



# System dependencies (required to compile C++ bindings)
- CMake >= 3.15
- MinGW-w64 or MSVC (Windows)
- Python 3.11–3.13 with 'Development' headers
# 1️⃣ Clone and enter project
git clone https://gitlab.com/your-team/Learned-Index-Structures.git
cd Learned-Index-Structures

# 2️⃣ Create and activate venv
python -m venv .venv
.venv\Scripts\activate  # (Windows)
# source .venv/bin/activate  # (Linux/macOS)

# 3️⃣ Install Python dependencies
pip install -r requirements.txt

# 4️⃣ Build C++ extension (PyBind11)
mkdir build && cd build
cmake ..
cmake --build . --config Release

# 5️⃣ Run main benchmark
cd ..
python main.py

