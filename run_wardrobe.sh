#!/bin/bash

SESSION="wardrobe"
BACKEND_DIR="."     # e.g., where your FastAPI `main.py` lives
FRONTEND_DIR="."   # e.g., where your Streamlit `app.py` lives

module use $HOME/MyModules
module load miniconda3/latest
conda activate multi_rag
# Start new session
tmux new-session -d -s $SESSION -n fastapi

# Run FastAPI in the first window
tmux send-keys -t $SESSION:fastapi "cd $BACKEND_DIR && uvicorn main:app --reload" C-m

# Create new window for Streamlit
tmux new-window -t $SESSION -n streamlit
tmux send-keys -t $SESSION:streamlit "cd $FRONTEND_DIR && streamlit run streamlit_app.py" C-m

# Optional: create a 3rd window for logs, shell, or monitoring
tmux new-window -t $SESSION -n shell
tmux send-keys -t $SESSION:shell "cd $FRONTEND_DIR" C-m

# Attach to the session
tmux attach -t $SESSION

