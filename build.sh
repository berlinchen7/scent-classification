# Install OS-agnostic requirements:
pip install -r requirements.txt

# Install OS-specific requirements:
case $1 in
    mac-cpu)
        # MPS acceleration is available on MacOS 12.3+
        pip3 install torch torchvision torchaudio
        pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
        ;;
    linux-cuda113)
        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
        pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
        ;;
    *)
        echo -n "Invalid OS option."
        echo 
        exit 1
        ;;
esac