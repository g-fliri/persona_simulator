# GenAI assignment

## Design and prototype a GenAI-based solution that creates realistic digital personas from human interviews.
Represent each human consumer using the interview data (see below for detailed data description)

The personas should be based on the content of the interview transcripts.
Enable each persona to respond to new test questions in a manner consistent with the original human profile.
The client’s goal is to mimic the real respondents as close as possible.
Evaluate how realistically these digital personas replicate human responses using quantitative and qualitative methods.
Design clear evaluation criteria for open-ended. Only evaluate open-ended questions since the client only cares about this question type in their market research. Evaluating the other question types is out of scope for the pitch.
Using the evaluation criteria, measure how closely the digital personas’ responses align with real human answers.

notes:
main.py needs an API key to make requests; create an .env file in the project directory with a variable OPENAI_API_KEY=<your_openAI_API_key>
run from bash 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to access MLflow from web server
project formatted with ruff
