today=`date +%Y-%m-%d.%H.%M.%S`


echo "Starting up fastapi application in the background. Give it a few seconds.."
cd ../src/
conda activate tf
# kill all processes on port 8000 to start fresh:
lsof -ti:8000 | xargs kill -9
python -m uvicorn dsif11app-fraud:app --reload --port 8504 &
UVICORN_PID=$! &
sleep 10

#echo "Starting up accumulator - will scan through transactions and score any new ones."
echo "Use ctrl + c to quit at any point"
#echo "In progress.. Logs will be written out to ../logs/automation_log_${today}.txt"
#mkdir -p ../logs/ # Create logs folder if not exist
streamlit run dsif11app-fraud-streamlit_test.py
STREAMLIT_PID=$!

# Trap to kill both processes when script is terminated
trap "kill $UVICORN_PID $STREAMLIT_PID" SIGINT

# Wait indefinitely to keep script running
wait