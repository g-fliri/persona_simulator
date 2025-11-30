# GenAI assignment

## Design and prototype a GenAI-based solution that creates realistic digital personas from human interviews.
Represent each human consumer using the interview data.

The personas should be based on the content of the interview transcripts.
Enable each persona to respond to new test questions in a manner consistent with the original human profile.
The client’s goal is to mimic the real respondents as close as possible.
Evaluate how realistically these digital personas replicate human responses using quantitative and qualitative methods.
Design clear evaluation criteria for open-ended. Only evaluate open-ended questions since the client only cares about this question type in their market research. Evaluating the other question types is out of scope for the pitch.
Using the evaluation criteria, measure how closely the digital personas’ responses align with real human answers.



## Run:

Type 'python main.py' from main folder to launch the server app. You can then open the frontend typing 'localhost:8808' in your web browser address bar. Ask questions and have fun!

Type 'python run_testset.py' to generate test runs on the excel dataset in 'data' folder'.

After running the testset, type 'python evaluate.py' to evaluate the generated replies against the expected answers in the 'data' folder. Evaluation metrics (cosine similarity and LLM-as-judge) are available in MLflow.

Run from bash 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to access MLflow from web server (address is available in the uvicorn console that automatically opens).

Altenatively, you can run from bash 'python mlflow_exp_summarizer.py' to have a (quite raw) comparison among experiments.



### Notes:

project formatted with ruff
